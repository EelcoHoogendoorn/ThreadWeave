

"""
context module
"""

verbose = False

import numpy as np



class AbstractContext(object):

    """
    mixes in backend-independent threadweave-specific functionality
    """

    def _kernel_declarations(self):
        """
        threadweave dependent extensions to threadpy
        """
        def cut():
            """cut contiguous region from one array to a smaller one"""
        def paste():
            """paste smaller image into a larger one"""

        def pad(arr, stencil, value):
            """
            create a new array, contaning arr, padded for access by stencil, with the padding filled with value
            """
            shape = tuple(zip( arr.shape, stencil.total_padding))
            padded = self.filled(shape, arr.dtype)



##        meshgrid = self.kernel("""
##            (<dtype>[2,i,j] grid) << [i,j] << ():
##                grid(0,i,j) = n;
##                grid(1,i,j) = m;
##            """)
##        def meshgrid():
##            pass
##        self.meshgrid = meshgrid



    #where the magic begins, from the POV of the end user
    def kernel(self, source):
        """flexible interface to kernel declaration. specify a source string or valid path to a source file"""
        import os.path
        if os.path.exists(source):
            source = open(source).read()
        from .frontend import parsing_declaration
        declaration, body = parsing_declaration.build_declaration(source)
        #body macro parsing; this should be a fallback option, if body fails to match DSL grammar
        from .frontend import parsing_macro
        declaration.body = parsing_macro.parse_body(body)
        return JustInTimeKernel(self, declaration)
    def tensor_product(self, source):
        """arbitrary tensor product; numpy.einsum work-alike"""
        from .frontend import parsing_tensor
        declaration = parsing_tensor.parse(source)
        return JustInTimeKernel(self, declaration)
    def elementwise_kernel(self, source):
        """
        source is a simple expression, to be applied to each element of each operand
        broadcasting rules may apply to the arguments
        returns a broadcasting-kernel-factory, which caches based on dimensionality
        """
        #note; we do not go from source to declaration here (rather, to factory first)
        #hence the different design than above
        return ElementwiseFactory(self, source)

    def binary_elementwise_kernel(self, op):
        """
        derived of elementwise; only needs a binary op as input
        """
        #note; we do not go from source to declaration here (rather, to factory first)
        #hence the different design than above
        return BinaryElementwiseFactory(self, op)


    def compile(self, declaration):
        """backend specific function, to transform a declaration into a kernel"""
        raise NotImplementedError()




class ElementwiseFactory(object):
    """
    created with expressions of the form '{0}+{1}'
    {index} is an argument dummy. all arguments must be broadcast-compatible

    all shapes are taken to be compile time constants
    different types, shapes, and thus dimensions as well, will all trigger a rebuild
    a change of type may influence output type in a non-obvious way, triggering a rebuild
    and for simplicity we recompile for all shape changes, since shape of 1 triggers
    broadcasting, which rquires special code emission

    alternatively, we can cache on a sig that tests if a dimension equals 1,
    which we take to be the broadcasting pattern. then we could default to making
    all non-broadcasted size args runtime variable. would require less compilation
    but per-thread initialization requirements would be rather hideous
    """
    def __init__(self, context, source):
        self.context = context

        from .frontend.parsing_elementwise import parse
        self.arguments, self.expression = parse(source)

        self.cache = {}

    def build(self, args):
        """
        build a declaration for this expresion, given the arguments
        this code need not be efficient, since it is behind cache
        """
        ndim = np.unique(arg.ndim for arg in args)
        assert(ndim.size==1)
        ndim = ndim[0]

        dtype = args[0].dtype.name       #to be replace with get_common_dtype equivalent. eval the given expression on zero size numpy arrays
        shapes = np.array([arg.shape for arg in args])
        shape = shapes.max(axis=0)

        #build a declaration
        from . import declaration
        decl = declaration.KernelDeclaration()

        #add axes to declaration
        from pyparsing import alphas
        axes = alphas[:ndim]
        for axis, size in zip(axes, shape):
            axis = decl.axis(identifier = axis, size = size)

        #add inputs
        for i, arg in enumerate(args):
            input = decl.input(identifier = 'input_{i}'.format(i=i), dtype=arg.dtype.name, shape = arg.shape)

        #add output
        output = decl.output(
                identifier = 'output',
                dtype = dtype,
                shape = shape)

        #broadcasting logic
        BC = shapes==1
        EQ = shapes == shape[np.newaxis,:]
        valid = np.logical_or(BC, EQ)
        assert(np.all(valid))           #assert valid broadcasting shape

        def arg_axes(arg, bc):
            return ','.join(a+'*0'*bca for a,bca in zip(axes, bc) )
        idx = [arg_axes(arg, bc) for arg, bc in zip(args, BC)]
        #construct broadcasting indexing expression
        body = 'output({}) = '.format(','.join(axes)) + self.expression.format(idx=idx) + ';'
        from .frontend import parsing_macro
        decl.body = parsing_macro.parse_body( body)

        return decl


    def instantiate(self, args):
        """
        caching based on full shapes and types
        implemented like this, because shapechange from bc to non-bc
        requires different code emission
        another simple implementation would be to pass in all strides, rather than computing internally
        """

        sig = tuple((arg.dtype, arg.shape) for arg in args)
        try:
            return self.cache[sig]
        except:
            declaration = self.build(args)
            kernel = JustInTimeKernel(self.context, declaration)
            self.cache[sig] = kernel
            return kernel


    def __call__(self, *args):
        """args is a list of input arguments"""
        #obtain a jit kernel
        instance = self.instantiate(args)
        #invoke the jit kernel
        return instance(*args)

class BinaryElementwiseFactory(ElementwiseFactory):
    """simple subclass of elementwise. needs to be inited with a single binary op only"""
    def __init__(self, context, op):
        super(BinaryElementwiseFactory, self).__init__(context, '{0}'+op+'{1}')


class JustInTimeKernel(object):
    """
    high level user visible kernel object
    wraps a templated kernel declaration
    and a cache of JITted kernel instantiations
    this is in no way backend specific
    rename to kernelfactory?
    """

    def __init__(self, context, declaration):
        self.context = context
        self.declaration = declaration
        self.cache = {}

    def instantiate(self, **templates):
        """
        instantiation of the templated kernel declaration is cached on signature of the template args
        thus, calling with the same compile-time constants neither leads to recompilation
        nor unnecessary code generation, while not as expensive, is also something youd rather do only once
        """
        sig = tuple(templates[key] for key in sorted(templates))    #hashable unique signature
        try:
            return self.cache[sig]
        except:
            instance = self.declaration.instantiate(templates)      #declaration instance
            kernel   = self.context.compile(instance)               #kernel instance
            self.cache[sig] = kernel
            return kernel

    def mine_template_arguments(self, args, templates):
        """
        mine calling arguments for unbound template parameters
        zip args and declaration
        if declaration's identifier is in the set of template identifiers
        append the value to the template dict
        """

        for v_arg, d_arg in zip(args, self.declaration.inputs):
            #mine type argument
            if d_arg.dtype in self.declaration.type_templates:
                templates[d_arg.dtype] = str(v_arg.dtype.name)
            #mine shape arguments
            for i, (v_size, d_axis) in enumerate(zip(v_arg.shape, d_arg.shape)):
                if not d_axis.identifier in self.declaration.size_variables:
                    if d_axis.identifier in self.declaration.size_templates:
##                        templates[d_axis.identifier] = v_size
                        if d_arg.boundary=='padded':
                            stencil = templates[d_arg.stencil]
                            padding = stencil.total_padding
                            templates[d_axis.identifier] = (v_size) - padding[i]
                        else:
                            templates[d_axis.identifier] = (v_size)


    def __call__(self, *args, **kwargs):
        #grab any directly specified template arguments
##        templates = kwargs.pop('templates', {})
        #mine JIT signature from args
        templates = {}
        for t in self.declaration.all_templates:
            if t in kwargs:
                templates[t] = kwargs.pop(t)
        self.mine_template_arguments(args, templates)

        #obtain a kernel instance for this set of template arguments
        instance = self.instantiate(**templates)
        #call the instance with the given arguments
        return instance(*args, **kwargs)

