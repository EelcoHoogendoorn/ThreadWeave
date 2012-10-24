
"""
CUDA module; all cuda subclasses go here
that is, context, kernel, and codegen
"""

import numpy as np

from pycuda.driver import init, Device
init()

from pycuda import gpuarray
from pycuda.compiler import SourceModule


from ..context import AbstractContext
import threadpy.backend.CUDA

class Context(AbstractContext, threadpy.backend.CUDA.Context):
    """CUDA context wrapper"""

    def __init__(self, device = 0, context = None):
        #init backend
        threadpy.backend.CUDA.Context.__init__(self, device, context)
        #init threadweave specific features
        AbstractContext.__init__(self)

    #device property accessors. only used by threadweave at the moment
    #but they are probably better abstracted away in threadpy
    #note that these are far from complete
    def supported_axes(self):
        cc = self.device.compute_capability()
        if cc < (2,0):
            return 'xy'     #only 2d threadblocks supported
        else:
            return 'xyz'
    def threads_per_block(self):
        cc = self.device.compute_capability()
        if cc < (2,0):
            return 512
        else:
            return 1024

    #perform CUDA compilation step on threadweave kernel declaration
    def compile(self, declaration):
        source = CodeGenerator(self, declaration).generate()
        module = SourceModule(source)
        kernel = module.get_function('kernel_main')
        return KernelInstance(self, declaration, kernel)



from ..instance import AbstractKernelInstance
class KernelInstance(AbstractKernelInstance):
    """wraps a raw pycuda kernel with array awareness, and grid/block policies"""

    def array_data(self, array):
        """grab from array data that info required to pass into kernel"""
        return array

    def grid_and_block(self, size_arguments):
        """
        size arguments is a dict of runtime args that complete the signature
        kindof hacky; ideally the binding of arguments should happen within the instance object
        otoh, performance is not a complete non-issue in this runtime code.
        """
        axes = [i for i,axis in enumerate(self.declaration.axes) if axis.is_parallel]

        def substitute(axis):
            return size_arguments.get( axis.identifier, axis.size)
        shape = tuple(substitute(axis) for axis in self.declaration.axes)

        leftover = self.context.threads_per_block()     #threads per block is device dependent; needs to come from context
        block = [1]*3
        #rather simple thread block assignment; assign as much as possible to slowest stride, and overflow into the next
        #but should work well under most circumstances.
        #this needs a manual override
        for i,a in enumerate( axes):
            block[i] = int( min(leftover, shape[a]))
            leftover = leftover //  block[i]
            if leftover == 0: break
        block = tuple(block)


        def iDivUp(a, b):
            # Round a / b to nearest higher integer value
            a = np.int32(a)
            b = np.int32(b)
            r = (a / b + 1) if (a % b != 0) else (a / b)
            return int(r)

        grid = tuple(iDivUp(shape[a],b) for b,a in zip(block, axes))


        return grid, block

    def invoke(self, arg_buffer, size_arguments):
        #do grid/block computations
        grid, block = self.grid_and_block(size_arguments)

        #invoke kernel
        self.kernel(*arg_buffer, grid = grid, block = block)




from ..code_generation import SIMD_Code_Generator
class CodeGenerator(SIMD_Code_Generator):
    """snippets that define cuda syntax"""
    def kernel_base(self):
        return '__global__ void kernel_main'

    def base_argument(self, arg):
        """declared argument"""
        return '{mutable}{type}{ptr} const {restrict}{identifier}'.format(
            mutable     = 'const ' * arg.immutable,
            type        = arg.dtype,
            ptr         = '*' * arg.is_array,
            restrict    = '__restrict__ ' * arg.is_array,
            identifier  = arg.identifier,
            )

    def parallel_axis_decl(self):
        return self.var_decl_statement('uint32', '{axis}', 'blockDim.{ha}*blockIdx.{ha} + threadIdx.{ha}')

    def atomic_op(self, op):
        return 'atomic{op}'.format(op=op.capitalize())

    synchronize_statement = '__syncthreads;'


def array_monkey_patch():
    """
    array behavior monkey patching; shouldnt there be a cleaner way of doing this?
    or perhaps al array subclassing should rather be foregone in favor of this?
    hmmm

    yet bigger problem; we require a context object ot build a kernel (so we can query underlying
    device properties for code generation). yet native arrays do not have a context attribute,
    instead probably relying on silly state machine bullshit. perhaps add cached context property to array?
    alternatively, we could require one context per process for use with threadweave
    but this is a tad hacky.
    """

    def wrap(orig_func):
        #http://downgra.de/2009/05/16/python-monkey-patching/
        """ decorator to wrap an existing method of a class.
            e.g.

            @wrap(Post.write)
            def verbose_write(forig, self):
                print 'generating post: %s (from: %s)' % (self.title,
                                                          self.filename)
                return forig(self)

            the first parameter of the new function is the the original,
            overwritten function ('forig').
        """
        import functools
        import inspect
        # har, some funky python magic NOW!
        @functools.wraps(orig_func)
        def outer(new_func):
            def wrapper(*args, **kwargs):
                return new_func(orig_func, *args, **kwargs)
            if inspect.ismethod(orig_func):
                setattr(orig_func.im_class, orig_func.__name__, wrapper)
            return wrapper
        return outer

    from threadpy.backend.CUDA import ndarray
    from pycuda.tools import context_dependent_memoize
    #broadcasting binary ops
    @context_dependent_memoize
    def badd():
        pass

    @wrap(ndarray.__add__)
    def __badd__(forig, self, other):
        try:
            return self + other
        else:
            pass

##array_monkey_patch()