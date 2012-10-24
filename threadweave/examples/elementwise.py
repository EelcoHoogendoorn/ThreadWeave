

"""
this test showcases the threadweave elementwise capability
by building on the nd-awareness of threadweave, the elementwise operations
in threadweave can easily be extended to allow for broadcasting operation
(and operations between arbitrarily strided arrays, once this is supported at the ndarray level)

TODO: overload operators in ndarray class
"""

from threadweave.backend import CUDA as Backend
import numpy as np


with Backend.Context(device=0) as ctx:
    #allocate some simple test arrays
    a = ctx.arange(4,  dtype=np.float32).reshape((4,1))
    b = ctx.arange(4,  dtype=np.float32).reshape((1,4))
    c = ctx.arange(16, dtype=np.float32).reshape((4,4))

    #elementwise kernels can be defines using these simple expressions
    #involving curly-braced argument positions
    broadcasting_product = ctx.elementwise_kernel('{0}*{1}')
    print broadcasting_product(a,b)

    #there is derived helper functionality for binary operations, specifically
    #these should be tacked onto the nd-array class
    broadcasting_sum = ctx.binary_elementwise_kernel('+')
    print broadcasting_sum(a,b)

    #but we can also take it in the other direction, of more compicated expressions
    funky_expression = ctx.elementwise_kernel('{0}*{1}+{2}')
    print funky_expression(a, b, c)

