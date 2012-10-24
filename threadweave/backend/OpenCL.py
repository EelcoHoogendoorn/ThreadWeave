"""
openCL backend module

note that I have never used any openCL until i started writing this module
so do not be surprised if this is riddled with bugs and suboptimal design

note that we have only 200 lines of backend-specific code, and it
would be closer to 100 with some more agressive abstraction away to base classes
"""
import numpy as np
import pyopencl as cl

import pyopencl.array


from ..context import AbstractContext
class Context(AbstractContext):

    def __init__(self, device = 0):
##        self.device = Device(device)
##        self.context = self.device.make_context()
        #ehhh, how to select a device in opencl?
        self.context = cl.create_some_context()
        self.queue = cl.CommandQueue(self.context)

        #dynamicaly compile and add kernel functions to class
        self._kernel_declarations()

    def __enter__(self):
        return self
    def __exit__(self, type, value, traceback):
       pass     #does openCL need any cleanup?



    #havnt figured out how to query device for these things yet...
    def supported_axes(self):
        return range(3)
    def threads_per_block(self):
        return 512

    #perform compilation step; compile declaration object to backend-specific kernel object
    def compile(self, declaration):
        source = CodeGenerator(self, declaration).generate()
        program = cl.Program(self.context, source).build()
        kernel = getattr(program, 'kernel_main')
        return KernelInstance(self, declaration, kernel)


    #array creation functions
    def array(self, object):
        if isinstance(object, np.ndarray):
            return pyopencl.array.to_device(self.queue, object)
        if isinstance(object, pyopencl.array.Array):
            arr = pyopencl.array.empty_like(object)
            pyopencl.array.Array._copy(arr, object)
            return arr
    def empty(self, shape, dtype=np.float32):
        return pyopencl.array.Array(self.queue, shape, dtype)
    def filled(self, shape, dtype=np.float32, fill = None):
        arr = self.empty(shape, dtype)
        if not fill is None: arr.fill(fill)
        return arr





from ..instance import AbstractKernelInstance
class KernelInstance(AbstractKernelInstance):
    """openCL specifics"""

    def array_data(self, array):
        #opencl arrays are passed in as buffers
        return array.data


    def grid_and_block(self, size_arguments):
        """size arguments is a dict of runtime args that complete the signature"""
        axes = [i for i,axis in enumerate(self.declaration.axes) if axis.is_parallel]

        def substitute(axis):
            return size_arguments.get( axis.identifier, axis.size)
        shape = tuple(substitute(axis) for axis in self.declaration.axes)

        leftover = self.context.threads_per_block()
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
        self.kernel(self.context.queue, grid, block, *arg_buffer, g_times_l=True)




from ..code_generation import SIMD_Code_Generator
class CodeGenerator(SIMD_Code_Generator):
    """snippets that define OpenCL syntax"""

    def kernel_base(self):
        return '__kernel void kernel_main'

    def base_argument(self, arg):
        return '__global {mutable}{type}{ptr} const {identifier}'.format(
            mutable     = 'const ' if arg.immutable else 'volatile ',
            type        = arg.dtype,
            ptr         = '*' * arg.is_array,
            identifier  = arg.identifier,
            )

    def parallel_axis_decl(self):
        return self.var_decl_statement('uint32', '{axis}', 'get_global_id({ha})')

    def atomic_op(self, op):
        """
        should we seek to do something about the rather shocking lack of opencl atomic_add(float) support,
        we should probably do it here
        """
        return 'atomic_{op}'.format(op=op.lower())

    synchronize_statement = 'barrier();'    #not used actually; just an example




