

"""
kernel instance module

instantiated and compiled version of a kernel

this module defines the abstract base class; much of the kernel is backend specific, however

"""

import numpy as np



class AbstractKernelInstance(object):
    """
    wraps a declaration and its derived backend-specific kernel object
    """
    def __init__(self, context, declaration, kernel):
        self.context = context
        self.declaration = declaration
        self.kernel = kernel

    def __call__(self, *args):
        raise NotImplementedError()

    def array_data(self, array):
        raise NotImplementedError()

    def check_arguments(self, args):
        """runtime argument validity checking"""
        #zip args and declaration
        for v_arg, d_arg in zip(args, self.declaration.arguments.itervalues()): #arguments iterated in order they were declared
            if d_arg.is_array:
                #check dtype
                assert(v_arg.dtype == d_arg.dtype)
                #check dims
                assert(len(v_arg.shape) == d_arg.ndim)
                #check size
                for i, (v_size, d_axis) in enumerate( zip(v_arg.shape, d_arg.shape)):
                    if d_axis.is_constant:
                        if d_arg.boundary=='padded':
                            stencil = self.declaration.get_stencil(d_arg.stencil)
                            assert(v_size == d_axis.size + stencil.total_padding[i])
                        else:
                            assert(v_size == d_axis.size)


    def mine_arguments(self, args):
        """
        mine the argument list for runtime-arguments

        abstract this stuff away in decl?
        """

        from collections import OrderedDict
        _size_arguments = {}    #temp dict

        #zip args and declaration
        for v_arg, d_arg in zip(args, self.declaration.arguments.itervalues()): #arguments iterated in order they were declared
            #mine shape arguments
            for i, (v_axis, d_axis) in enumerate(zip(v_arg.shape, d_arg.shape)):
                if d_axis.is_variable:
                    cast = np.uint32
                    if d_arg.boundary=='padded':
                        stencil = self.declaration.get_stencil(d_arg.stencil)
                        padding = stencil.total_padding
                        _size_arguments[d_axis.identifier] = cast(v_axis) - padding[i]
                    else:
                        _size_arguments[d_axis.identifier] = cast(v_axis)

        #reorder size arguments in order as specified by size_arguments set (same as used by code generator to create C signature)
        size_arguments = OrderedDict()
        for axis in self.declaration.size_variables:
            size_arguments[axis] = _size_arguments[axis]

        return size_arguments

    def base_arguments(self, args):
        """
        given complete list of input/output args
        build list of c-ready information
        """

        def base(decl, value):
            """arrays are passed in in background dependent way. scalars are passed in as numpy types"""
            if decl.is_array:
                return self.array_data(value)
            if decl.is_scalar:
                cast = getattr(np, decl.dtype)
                return cast(value)

        return [base(decl, value) for  value, decl in  zip(args, self.declaration.arguments.itervalues())]

    def output_arguments(self, args, size_arguments):
        """
        allocate missing output args,
        and build list of returned objects
        """
        def substitute(axis):
            return size_arguments.get( axis.identifier, axis.size)

        output_arguments = []    #list of things to return from kernel

        outputs = iter(self.declaration.outputs)
        #simple copy over supplied output args to output list; consume a matching number of output declarations
        for value, decl in zip(args[len(self.declaration.inputs):], outputs):
            output_arguments.append(value)
        #allocate the missing arguments
        for output in outputs:
            shape = tuple(int(substitute(axis)) for axis in output.shape)  #deref to size
            if output.default in self.declaration.input_identifiers:
                #init output with copy of input
                index = self.declaration.input_identifiers.index(output.default)
                value = self.context.array(args[index])
            else:
                #allocate output, and fill with default, if available
                if output.boundary=='padded':
                    #adjust shape to padding of stencil
                    stencil = self.declaration.get_stencil(output.stencil)
                    shape = tuple(s+p for s,p in zip(shape, stencil.total_padding))
                #allocate
                if output.default:
                    value = self.context.filled(shape, output.dtype, output.default)
                else:
                    value = self.context.empty(shape, output.dtype)
            #append to list of calling and return arguments
            args.append(value)
            output_arguments.append(value)

        return args, output_arguments



    def __call__(self, *args, **kwargs):
        """
        invoke the kernel
        this involves
         - checking arguments for correctness
         - collecting runtime size arguments from the argument list
         - allocating missing output arguments
         - performing grid and block calculations
         - invoking the backend-specific kernel
        and
        """
        args = list(args)

        #check arguments for compatibility with the declaration
        self.check_arguments(args)

        #pull runtime variables from the argument list
        size_arguments = self.mine_arguments(args)
        for s in self.declaration.size_variables:
            if s in kwargs:
                size_arguments[s] = kwargs.pop(s)

        #allocate output arguments; extend args list
        args, output_args = self.output_arguments(args, size_arguments)

        #build total output buffer
        arg_buffer = (
            self.base_arguments(args) +
            [np.uint32(s) for s in size_arguments.values()])

        #invoke the kernel; backend specific; give size arguments to be able to do thread/block logic
        self.invoke(arg_buffer, size_arguments)

        #return the output args
        if len(output_args)==1:
            return output_args[0]
        else:
            return tuple(output_args)


