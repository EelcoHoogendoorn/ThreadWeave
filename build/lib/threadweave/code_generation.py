
"""
code emission backend; transform kernel declaration to code

so far, we have basic CUDA and openCL support
"""

verbose = True

from .frontend.parsing import indent, dedent

class CodeGenerator(object):
    """
    base class for code generation
    defines some common behavior
    but mostly what it does, is stringing together subclass behaviors
    """
    def __init__(self, context, declaration):
        self.context = context
        self.declaration = declaration

    def generate(self):
        """
        code generation happens here
        """

        #call declaration body functor with code generator, to obtain body string
        body = self.declaration.body(self)

        #wrap body in serial axes code
        for axis in self.declaration.serial_axes:
            body = self.serial_axis(body, axis)
        #wrap body in parallel axes code
        for axis, hardware_axis in zip(self.declaration.parallel_axes, self.context.supported_axes()):
            body = self.parallel_axis(body, axis, hardware_axis)


        #global composition
        arguments = self.runtime_arguments()
        init = self.initialization()
        source = self.function(arguments, init, body)

        #replace typing to c-type; body is already done,
        from .frontend import parsing_macro
        source = parsing_macro.replace_typing(self, source)

        if verbose:
            print source

        return source




    #abstract interface for a code generator (outdated)
##    def signature(self, declaration):                       raise NotImplementedError()
##    def array_arguments(self, array_declaration):           raise NotImplementedError() #mem ptr and variable shapes
##    def array_constants(self, array_declaration):           raise NotImplementedError() #fixed shapes and strides
##    def array_dependents(self, array_declaration):          raise NotImplementedError() #shapes linked to other args
##    def array_computants(self, array_declaration):          raise NotImplementedError() #shapes linked to other args
##
##
##    def function(self, arguments, init, body):              raise NotImplementedError()
##    def parallel_axis(self,body, axis, hardware_axis):   raise NotImplementedError()
##    def serial_axis  (self,body, axis):                  raise NotImplementedError()






import os
import pycuda.tools
dtype_to_ctype = {k:v for k,v in pycuda.compyte.dtypes.DTYPE_TO_NAME.iteritems() if isinstance(k, str)}

class C_CodeGenerator(CodeGenerator):
    """
    define stuff common to C dialects here;
    """


    #some utility methods for composing multiple pieces of code
    def arguments(self, arguments, newline=False):
        if newline:
            base = ', \n'
        else:
            base = ', '
        return base.join(arguments)
    def statements(self, statements, newline=True):
        if newline:
            base = '\n'
        else:
            base = ' '
        return base.join(statements)
    def paragraphs(self, paragraphs):
        return '\n\n'.join(paragraphs)
    def scope(self, body, newline=True):
        n = os.linesep * newline
        return '{'+n+indent(body)+n+'}'+n
    def list(self, body, newline = False):
        if newline:
            n = '\n'
        else:
            n = ''
        return '('+n+indent(body)+')'+n



    #body code snippets
    #this can be used either in a macro system or a proper DSL
    #either way, these are all plain-old C
    def for_statement(self, start, stop, step, index):
        return 'for (int {index}={start}; {index} < {stop}; {index}+={step})'.format(start=start, stop=stop, step=step, index=index)

    def indexing_expression(self, arg, indices):
        #select the right indexing method for the specified boundary handling
        index_expr = dict(
            padded = self.direct_index,
            clamped = 0,
            wrapped = self.wrapped_index,
            fixed = 0,
            none = self.direct_index,
            )[arg.boundary]

        offset = '+'.join(index_expr(arg, dim, index) for dim,index in enumerate(indices))\
                    +'+'+'{arr_id}_offset'.format(arr_id=arg.identifier)
        return '{identifier}[{offset}]'.format(identifier=arg.identifier, offset=offset)

    def direct_index(self, arg, dim, index):
        return '{arr_id}_stride_{dim}*{idx}'.format(
            arr_id  = arg.identifier,
            dim  = dim,
            idx  = index)
    def wrapped_index(self, arg, dim, index):
        return '{arr_id}_stride_{dim}*({idx}%{arr_id}_shape_{dim})'.format(
            arr_id  = arg.identifier,
            dim  = dim,
            idx  = index)






    def type_specifier(self, dtype):
        try:
            return dtype_to_ctype[dtype]
        except:
            raise

    def var_decl_expr(self, dtype, identifier, immutable = True):
        return '{type}{immutable} {identifier}'.format(
                immutable  = ' const' * immutable,
                type       = self.type_specifier(dtype),
                identifier = identifier)

    def var_decl_statement(self, dtype, identifier, rhs = None, immutable = True):
        expr = self.var_decl_expr(dtype, identifier, immutable)
        if rhs:
            return expr + ' = {rhs};'.format(rhs=rhs)
        else:
            return expr + ';'






    #compile-time and run-time function arguments
    def runtime_arguments(self):
        base_arguments = self.base_arguments()
        size_arguments = self.size_arguments()
        if size_arguments:
            return self.arguments([base_arguments, size_arguments], newline=True)
        else:
            return base_arguments
    def base_arguments(self):
        return self.arguments(
            self.base_argument(arg) for arg in self.declaration.arguments.itervalues())
    def size_arguments(self):
        return self.arguments(
            self.size_argument(var) for var in self.declaration.size_variables)
    def size_argument(self, var):
        return self.var_decl_expr('uint32', '{axis}_size'.format(axis = var))

    def initialization(self):
        return self.paragraphs([
            self.init_kernel(),
            self.init_arrays(),
            ])

    def init_kernel(self):
        return self.statements(
            self.init_kernel_axis(axis) for axis in self.declaration.axes if not axis.runtime_variable)
    def init_kernel_axis(self, axis):
        return self.var_decl_statement('uint32', axis.identifier+'_size', axis.size)

    def init_arrays(self):
        return self.paragraphs(
            self.init_array(arg) for arg in self.declaration.arguments.values() if arg.is_array)
    def init_array(self, arg):
        """shapes and strides for a given array"""
        return self.statements([
                self.init_array_shapes(arg),
                self.init_array_strides(arg),
            ])


    def init_array_shapes(self, arg):
        return self.statements(
            self.init_array_shape(arg.identifier, i, axis) for i, axis in enumerate(arg.shape))
    def init_array_shape(self, identifier, dimension, axis):
        """assign array shape constsants of the form {array}_shape_{dimension}, with either runtime or compile time expression"""
        identifier = '{array}_shape_{dimension}'.format(array=identifier, dimension=dimension)
        rhs = axis.size if axis.is_constant else '{axis}_size'.format(axis=axis.identifier)
        return self.var_decl_statement('uint32', identifier, rhs)


    def init_array_strides(self, arg):
        if arg.boundary=='padded':
            stencil = self.declaration.get_stencil(arg.stencil)
        else:
            stencil = None
        return self.statements(self.init_array_strides_generator(arg, stencil))
    def init_array_strides_generator(self, arg, stencil):
        identifier, shape = arg.identifier, arg.shape
        ndim = len(shape)
        if stencil is None:
            padding = [0] * ndim
            left_padding = padding
        else:
            padding = stencil.total_padding
            left_padding = stencil.left_padding

        #some commonly used constant expressions
        def decl(identifier, value):
            return self.var_decl_statement('uint32', identifier, value)
        stride_template  = '{identifier}_stride_{dimension}'
        shape_template = '{identifier}_shape_{dimension}'

        #yield stride declarations
        prev = stride_template.format(identifier=identifier, dimension=ndim-1)
        yield decl(prev, 1)
        for i, (size, pad) in reversed(list(enumerate( zip( shape, padding)[:-1]))):
            this = stride_template.format(identifier=identifier, dimension=i)
            size = shape_template.format(identifier=identifier, dimension=i+1)
            stride_value = '{prev} * ({size}+{padding})'.format(prev=prev ,size=size, padding=pad)
            yield decl(this, stride_value)
            prev = this

        #add total element size as well, for good measure
        size = shape_template.format(identifier=identifier, dimension=0)
        size_value = '{prev} * ({size}+{padding})'.format(prev=prev ,size=size, padding=padding[0])
        size_identifier = '{identifier}_size'.format(identifier=identifier)
        yield decl(size_identifier, size_value)

        #offset of first element from base pointer (nonzero for padded arrays)
        o_template  = '({identifier}_stride_{dimension} * {padding})'
        offset_value = '+'.join(
                o_template.format(identifier=identifier, dimension=i, padding=p)
                    for i,p in enumerate(left_padding))
        offset_identifier = '{identifier}_offset'.format(identifier=identifier)
        yield decl(offset_identifier, offset_value)






    #global function architecture
    # we have arguments, a section of pre-loop initalizations (stride computations, f.i.)
    #and then a body, wrapped in several layers of axis looping
    def function(self, arguments, init, body):
        return self.statements([
            self.kernel_base(),
            self.list(arguments, True),
            self.scope(self.paragraphs([init, body]))
            ])
    def parallel_axis(self, body, axis, hardware_axis):
        return \
            self.paragraphs([
                self.statements([
                    '//launch threads for parallel axis',
                    self.parallel_axis_decl(),
                    'if ({axis} >= {axis}_size) return;']).format(axis = axis.identifier, ha = hardware_axis),
                body])
    def serial_axis(self, body, axis):
        return \
            self.paragraphs([
                self.statements([
                    '//loop over serial axis',
                    'unsigned {axis};',
                    'for ({axis}=0; {axis} < {axis}_size; {axis}++ )']).format(axis=axis.identifier),
                self.scope(body)])


class BlitzCodeGenerator(C_CodeGenerator):
    """
    just an idea so far, but should be easy to realize, no?
    given how much blitz functionality we reimplement, might choose something more lightweight?
    """

class SIMD_Code_Generator(C_CodeGenerator):
    """
    idioms common to opencl and cuda go here
    is there anything shared between opencl and cuda that differs from C though?
    not much, probably
    """
