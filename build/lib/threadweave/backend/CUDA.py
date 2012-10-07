
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
class Context(AbstractContext):
    """CUDA context wrapper"""

    def __init__(self, device = 0, context = None):
        if context is None:
            #create context on specified device
            self.device = Device(device)
            self.context = self.device.make_context()
        else:
            #init from existing context; merely act as wrapper
            self.context = context
        self._kernel_declarations()

    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
       self.context.pop()   #needed for proper cleanup


    def supported_axes(self):
        """
        example of abstracting away some device specifics
        this one happens to be relevant between my laptop and desktop
        but there is probably tons more of these kind of things i havnt thought about
        """
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

    #perform compilation step
    def compile(self, declaration):
        source = CodeGenerator(self, declaration).generate()
        module = SourceModule(source)
        kernel = module.get_function('kernel_main')
        return KernelInstance(self, declaration, kernel)

    #array creation functions
    def array(self, object):
        if isinstance(object, np.ndarray):
            return gpuarray.to_gpu(np.array(object))
        if isinstance(object, gpuarray.GPUArray):
            return object * 1
    def empty(self, shape, dtype=np.float32):
        return gpuarray.empty(shape, dtype)
    def filled(self, shape, dtype=np.float32, fill = None):
        arr = self.empty(shape, dtype)
        if not fill is None: arr.fill(fill)
        return arr





from ..instance import AbstractKernelInstance
class KernelInstance(AbstractKernelInstance):
    """wraps a raw pycuda kernel with array awareness"""

    def array_data(self, array):
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

