"""
defines the stencil class for use with stencil syntax,
plus constructors for / instances of some commonly used stencils

for simplicity, a stencil is always treated as compile time constant, and are unrolled
really huge kernels might be better off not being unrolled.
I do not really have a usecase for such atm,
but should not be too hard to implement this feature
"""

import numpy as np


class Voxel(object):
    """stencil voxel helper class"""
    def __init__(self, offset, weight):
        self.offset = tuple(offset)
        self.weight = weight


class Stencil(object):
    def __init__(self, weights, mask=0, origin = None):
        self.weights = weights
        if origin is None:
            #by default, stencils are centered
            self.origin = tuple(i//2 for i in self.shape)
        else:
            self.origin = origin

        if mask is None:
            self.mask = np.zeros_like(weight)
        elif isinstance(mask, np.ndarray):
            self.mask = mask
        else:
            self.mask = weights==mask

        assert(self.mask.shape==self.weights.shape)

    def __hash__(self):
        return id(self)

    @property
    def shape(self): return self.weights.shape
    @property
    def ndim(self): return len(self.shape)

    @property
    def size(self):
        return np.count_nonzero(self.mask==0)

    @property
    def total_padding(self):
        return tuple(s-1 for s in self.shape)
    #start/stop padding?
    @property
    def left_padding(self):
        return self.origin
    @property
    def right_padding(self):
        return tuple(t-l for t,l in zip(self.total_padding, self.left_padding))

    def voxels(self):
        """unrol the stencil into a list of voxel objects"""
        d = distance_vector(self.shape, self.origin)
        return [
            Voxel(v, w)
                for v,w,m in zip( d.reshape((-1,self.ndim)), self.weights.flatten(), self.mask.flatten())
                    if not m]

    def __getitem__(self, idx):
        """note; returned stencil does not preserve origin properties of parent!"""
        return Stencil(self.weights[idx], self.mask[idx])



def distance_vector(shape, origin=None):
    """a meshgrid function, centered at origin"""
    if origin is None:
        origin = tuple(i//2 for i in shape)
    slcs = [slice(-o,s-o) for s,o in (zip( shape, origin))]
    return np.transpose ( np.mgrid[slcs])

def distance(vec):
    return np.sqrt((vec**2).sum(axis=-1))

def sphere(dims, max_r, min_r=1):
    """
    spherical stencil factory
    contains all voxels with a distance from the origin between the specified radii
    can be used to generate a great number of common stencils:
        sphere(2,1)   -> 4 nearest neighbors
        sphere(2,1.5) -> full ring of 8 neighbors
        sphere(3,1)   -> 6 nearest neighbors
        sphere(2,1.5,0) ->3x3 full stencil, for averaging filter f.i
    """
    shape = [int(max_r)*2+1]*dims
    d = distance(distance_vector(shape))
    mask = np.logical_or(d>max_r, d<min_r) * 1
    return Stencil(d, mask)

laplacian_1_3 = Stencil(np.array([ 1, -2, 1]))

laplacian_2_5 = Stencil(
    np.array([
        [ 0, 1, 0],
        [ 1,-4, 1],
        [ 0, 1, 0],
    ]))

laplacian_2_9 = Stencil(
    np.array([
        [  1,  4,  1],
        [  4,-20,  4],
        [  1,  1,  1],
    ]).astype(np.float) / 6)

laplacian_3_7 = Stencil(
    np.array([
        [
            [ 0, 0, 0],
            [ 0, 1, 0],
            [ 0, 0, 0],
        ],
        [
            [ 0, 1, 0],
            [ 1,-6, 1],
            [ 0, 1, 0],
        ],
        [
            [ 0, 0, 0],
            [ 0, 1, 0],
            [ 0, 0, 0],
        ],
    ]))



def laplacian(dims, support):
    """parametrized accessor for some predefined laplacian stencils"""
    try:
        return globals()['laplacian_{dims}_{support}'.format(dims=dims,support=support)]
    except:
        raise Exception('specified laplacian not defined')

def padding(shape, origin = None):
    """create a dummy stencil, with the sole intended use of specify padding for an array"""
    return Stencil(np.ones(shape), origin = origin)