
"""
declarations API and datamodel

this declaration is what defines the kernel
it is used by the code generator to build up output code
and it is used by the runtime to check for argument correctness

this declaration structure can either be built up 'manually'
but the preferred way of usage is by specifying a declaration by means
of a special DSL
the parsing_declaration and parsing_tensor modules provide examples of such




if there ever was a piece of pyhon code that should emphasize clarity
over performance, this would be it

this module could do with another round of reorganization

would be good to split code in a mutable declarationfactory,
what we use to build up a declaration object,
and then 'compile' it into an immutable declaration object,
where all relations between parts are computed during initialization

instantiating a declaration with a templates dict should idealy build
a whole new object hierarchy, where all template arguments are bound,
and which presents a slick interface for maintaining and binding the
runtime arguments

template or runtime variables should be their own objects, rather than
being implicitly defined by missing values, as is currently happening
this would lead to much cleaner code


"""

##class ArgumentDeclarationFactory(object):
##    def __init__(self, dtype, axes, identifier, default = None):
##        self.dtype = dtype
##        self.axes = axes
##        self.identifier = identifier
##        self.default = default
##
##class KernelDeclarationFactory(object):
##    def __init__(self):
##        self.inputs = []
##        self.outputs = []
##        self.axes = []
##
##    def __enter__(self):
##        pass
##
##    def __exit__(self, type, value, traceback):
##        self.finalize()
##
##    def finalize(self):
##        """
##        signature should be complete now; do hookup
##        and create actual declaration object
##        """
##
##    #factory methods
##    def Input(self, dtype, axes, identifier):
##        self.inputs.append(ArgumentDeclarationFactory(dtype, axes, identifier))
##    def Output(self, dtype, axes, identifier, default = None):
##        self.inputs.append(ArgumentDeclarationFactory(dtype, axes, identifier, default))



from .frontend.parsing import dtypes
from collections import OrderedDict



class KernelDeclaration(object):

    def __init__(self):
        self.body = None     #callable that generates a body text upon binding of a code generator
        self.axes = []       #list of axes declarations
        self.inputs = []     #list of input declarations
        self.outputs = []
        self.arguments = OrderedDict()  #dict of all arguments, keyed by identifier. duplicate id inout args are tagged as such

        self.size_constants = {}
        self.type_templates  = set()   #set of unbound type arguments
        self.size_templates = set()   #set of unbound size arguments
        self.size_variables = set()   #set of runtime variable shape arguments

        self.stencil_templates = set()     #set of unbound stencil arguments


    #factory methods for elements of the declaration
    def axis(self, **kwargs):
        axis = AxisDeclaration(self, **kwargs)
        if axis.is_variable:
            self.size_templates.add(axis.identifier)
        if axis.is_constant:
            self.size_constants[axis.identifier] = axis.size
        self.axes.append(axis)
        return axis
    def input(self, **kwargs):
        kwargs['writeable'] = False
        kwargs['readable'] = True
        arg = ArgumentDeclaration(self, **kwargs)
        self.inputs.append(arg)
        self._add_argument(arg)
        return arg
    def output(self, **kwargs):
        kwargs['writeable'] = True
        kwargs['readable'] = False
        arg = ArgumentDeclaration(self, **kwargs)
        self.outputs.append(arg)
        self._add_argument(arg)
        return arg

    #internal functions; should we do all this upon closing the context?
    def _add_argument(self, arg):
        #mine type templates
        self._add_type_template(arg)
        #mine size templates
        for axis in arg.shape:
            self._add_size_template(axis)
        #register arg in correct order
        self.arguments[arg.identifier] = arg

    def _add_type_template(self, arg):
        if not arg.dtype in dtypes:
            self.type_templates.add(arg.dtype)
    def _add_size_template(self, axis):
        if axis.is_variable:
            self.size_templates.add(axis.identifier)

    def get_stencil(self, stencil):
        return self.stencils[stencil]

    #accessor properties
    @property
    def axes_identifiers(self): return [axis.identifier for axis in self.axes]
    @property
    def serial_axes(self): return [axis for axis in self.axes if axis.is_serial]
    @property
    def parallel_axes(self): return [axis for axis in self.axes if axis.is_parallel]
    @property
    def array_identifiers(self): return [arg.identifier for arg in self.arguments.itervalues() if arg.is_array]
    @property
    def input_identifiers(self): return [arg.identifier for arg in self.inputs]
    @property
    def arrays(self): return [arg for arg in self.arguments.itervalues() if arg.is_array]

    @property
    def all_templates(self): return set.union( self.size_templates, self.type_templates, self.stencil_templates)

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def instantiate(self, templates):
        """
        create an instantiated copy of the templated declaration
        ensure all type args are bound
        unbound shape args become variable args
        """
        instance = self.copy()

        #make sure to substitute known constants into all possible locations
        templates.update(self.size_constants)

        #make substitutions in all subcomponents of the declaration
        for axis in instance.axes:
            axis.instantiate(templates)
        for input in instance.inputs:
            input.instantiate(templates)
        for output in instance.outputs:
            output.instantiate(templates)

        #check that we bound all type templates
        assert(all(dtype in templates for dtype in self.type_templates))
        #instance.bound_types = {ttype: templates[ttype] for ttype in self.type_templates}

        #append unbound shape templates to shape variables
        instance.size_variables.update(
            set(size for size in self.size_templates if not size in templates))

        #mine stencil template args
        instance.stencils = {stencil: templates[stencil] for stencil in self.stencil_templates }

        instance.templates = templates  #remember templates that define the kernel
        return instance



class ArgumentDeclaration(object):

    """
    usage:
        only to be constructed through kernel factory methods

        required arguments:
            identifier: valid identifier string
            dtype:      either a numpy dtype string, or a valid template identifier
        optional arguments:
            shape:      tuple of integers or axis identifiers, from which the shape is implied. if not specified, scalar argument is implied
            readable:   defaults to true; can be set to false for write-only memory
            writeable:  defaults to true; can be set to false for read-only memory (inout arg is both readable and writeable)
    """
    def __init__(self, kernel, **kwargs):
        self.kernel = kernel

        self.identifier = kwargs.pop('identifier')
        self.dtype      = kwargs.pop('dtype')

        self.shape      = tuple(DimensionDeclaration(axis) for axis in kwargs.pop('shape', ()))

        self.boundary   = 'none'        #actually, we might want to set seperate boundary handling for each axis. cant think of a use case though, and it sure would clutter syntax


        self.readable   = kwargs.pop('readable', True)
        self.writeable  = kwargs.pop('writeable', True)

        self.default    = kwargs.pop('default', None)
        if self.default == '': self.default = None

    def set_boundary(self, mode, value):
        self.boundary = mode
        if self.boundary=='padded':
            self.stencil = value
        if self.boundary=='fixed':
            self.fixedvalue = value

    def instantiate(self, templates):
        for axis in self.shape:
            axis.instantiate(templates)
        self.dtype = templates.get(self.dtype, self.dtype)


    @property
    def is_scalar(self): return self.ndim is 0
    @property
    def is_array(self): return not self.is_scalar
    @property
    def ndim(self): return len(self.shape)
    @property
    def immutable(self): return not self.writeable

    @property
    def padding(self):
        assert(self.boundary=='padded')
        stencil = self.kernel.get_stencil(self.stencil)
        return stencil.left_padding




class AxisDeclaration(object):


    def __init__(self, kernel, **kwargs):
        self.kernel = kernel

        self.identifier = kwargs.pop('identifier')
        self.size       = kwargs.pop('size', None)
        if self.size=='': self.size = None

        self.size_constraint = ''            #size constraint expresssion of the form low < identifier < high; evalled at runtime
#        self.parallel = True                 #parallel axis flag
        self.iteration = 'parallel'
        self.reduction = 'none'                #possible reduction operator bound to this axis? or how to handle this?

        self.runtime_variable = False               #flag to set a var to being runtime variable
        self.binding = 'run compile bound'      #need cleaner way to handle this
        #names need to be cleared up to explicitly distunguish compile time and run time arguments
        #axis size can have three states: constant, compile-time argument, and run-time argument
        assert(not (self.runtime_variable and self.is_constant))

    def instantiate(self, templates):
        if self.is_constant: return
        self.size = templates.get(self.identifier, None)

    @property
    def is_variable(self): return self.size is None
    @property
    def is_constant(self): return not self.is_variable
    @property
    def is_parallel(self): return self.iteration == 'parallel'
    @property
    def is_serial(self): return self.iteration == 'serial'
    @property
    def is_hybrid(self): return self.iteration == 'hybrid'

class DimensionDeclaration(object):
    """
    analogue of axes declaration
    but for an array, rather than kernel
    quite a bit of shared code; subclass?
    """

    def __init__(self, size):
        if isinstance(size, str):
            self.identifier = size
            self.size = None
        if isinstance(size, int):
            self.identifier = None
            self.size = size


        self.runtime_variable = False               #flag to set a var to being runtime variable
        #names need to be cleared up to explicitly distunguish compile time and run time arguments
        #axis size can have three states: constant, compile-time argument, and run-time argument
        assert(not (self.runtime_variable and self.is_constant))


    def instantiate(self, templates):
        if self.is_constant: return
        self.size = templates.get(self.identifier, None)

    @property
    def is_variable(self): return self.size is None
    @property
    def is_constant(self): return not self.is_variable







"""
body declaration interface
designed to enable different design patterns:
    either a macro preprocess transform based design
    or a proper langauge with parsing into tokens

with preprocessing, structure of code is implied
with language, the codegenerator needs to lay down the structure
"""
class AbstractBodyDeclaration(object):
    def __call__(self, code_generator):
        raise NotImplementedError()

class MacroBodyDeclaration(AbstractBodyDeclaration):
    """
    silly wrapper around functor
    """
    def __init__(self, functor):
        self.functor = functor
    def __call__(self, code_generator):
        self.functor(code_generator)

class DSLBodyDeclaration(AbstractBodyDeclaration):
    """
    body declaration explicitly built from tokens
    requires either a fairly complete C parser,
    or an DSL of equal expressiveness
    definitely on the TODO, but I cant quite oversee the scope of this

    the backend then needs to traverse this declaration to emit corresponding code
    """

    @staticmethod
    def from_pyparsing(parsed):
        self = KernelDeclaration()
        return self



if __name__ is '__main__':

    #some unit tests; example of manual declaration building
    decl = KernelDeclaration()

    decl.axis(identifier = 'i')
    decl.axis(identifier = 'j', size=10)

    decl.input(identifier = 'input_1', dtype='float32', shape = (2, 'i', 'j'))

    decl.body = MacroBodyDeclaration(lambda cg: """input_1(0,i,j) = i*j;""")







