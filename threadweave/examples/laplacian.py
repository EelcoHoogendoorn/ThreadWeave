import numpy as np
from threadweave.backend import CUDA as Backend
from threadweave.stencil import laplacian

with Backend.Context(device = 0) as ctx:

    laplacian_kernel = ctx.kernel("""
        (<type>[d,n,m] output) << [d,n,m] << (<type>[d,n,m] input):
            n:      serial
            d:      variable
            input:  padded(stencil)
        {
            <type> r = 0;
            for (s in stencil(input))
                 r += input(s.d,s.n,s.m) * s.weight;
            output(d,n,m) = r;
        }""")

    shape = (2,6,6)
    input = ctx.array(np.arange(np.prod(shape)).astype(np.float32).reshape(shape) ** 2)
    print input    #some arbitrary input data

    stencil = laplacian(2,5)[np.newaxis,:,:] #construct 2-dim 5-pts laplacian, with broadcasting
    output = laplacian_kernel(input, stencil = stencil)
    print output
