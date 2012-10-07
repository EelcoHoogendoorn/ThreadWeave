

"""
context module
"""

verbose = False

import numpy as np


class AbstractContext(object):

    """
    minimal wrapper for backend context
    allows abstracting away the backend behind a uniform interface
    """


    def __enter__(self):                        raise NotImplementedError()
    def __exit__(self, type, value, traceback): raise NotImplementedError()


    def _kernel_declarations(self):
        """
        add some numpy like functionality to the context, for arbitrary backend
        """

##        arange = self.kernel("""(<dtype>[n] arange) << [n] << (): arange(n) = n;""")
##        self.arange = lambda size, dtype: arange(templates=dict(dtype=np.dtype(dtype).name, n=size))
        self.arange = lambda size, dtype: self.array(np.arange(size, dtype=dtype))

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


    #array creation functions
    def array(self, object): raise NotImplementedError()
    def empty(self, shape, dtype=np.float32): raise NotImplementedError()
    def filled(self, shape, dtype=np.float32, fill = None): raise NotImplementedError()
    def ones(self, shape, dtype=np.float32):
        return self.filled(shape, dtype, 1)
    def zeros(self, shape, dtype=np.float32):
        return self.filled(shape, dtype, 0)
    def random(self, shape, dtype):
        return self.array(np.random.random(shape).astype(dtype) )

    #act on numpy arrays; not really part of interface
    #place numpy util functions yet somewhere else?
    def cut(self, source, S, E):
        slice(s, )
        return



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


    def compile(self, declaration): raise NotImplementedError()









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

