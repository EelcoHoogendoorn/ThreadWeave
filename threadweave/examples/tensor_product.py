import numpy as np
from threadweave.backend import CUDA as Backend

with Backend.Context(device = 0) as ctx:
    #some example data
    shape = 2,3,4
    A = ctx.arange(np.prod(shape),np.float32).reshape(shape)
    shape = 2,4,3
    B = ctx.arange(np.prod(shape),np.float32).reshape(shape)

    #declare product, contracted over j
    stacked_matrix_product = ctx.tensor_product('nij,njk->nik')
    C = stacked_matrix_product(A, B)

    print C
    print ctx.tensor_product('nii->ni')(C)   #print the diagonals of C too, for good measure
