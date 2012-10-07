import numpy as np
import matplotlib.pyplot as pp
from time import clock

from threadweave.backend import CUDA as Backend     #some parts fail for opencl, due to use of atomtic float adds
from threadweave.stencil import sphere

with Backend.Context(device = 0) as ctx:

    #averaging filter
    #init output with copy of input
    #this ensures copying over the padding, in case of padded arrays
    #wrapped access is implemented as integer modulus
    #this only works if the array shape is a power of two, in which case
    #CUDA replaces this with a bitwise and
    #which technically is not the same for negative numbers... bit hacky
    averaging = ctx.kernel(
        """
        (<type>[n,m] output = input) << [n,m] << (<type>[n,m] input):
            input:  wrapped
            output: wrapped
            n: serial
        {
            <type> r = 0;
            for (s in stencil(input))
                 r += input(s.n,s.m);
            output(n, m) = r / stencil.size;
        }
        """)
    stencil = sphere(2,1.5,0)   #3x3 constant stencil

    #construct smoothed random data
    value = ctx.random((256,256), np.float32)
    for i in xrange(5):
        value = averaging(value, stencil = stencil)

    #do some transforms on data, to make it more interesting for segmentation
    value -= value.get().min()
    value /= value.get().max()
    value **= 3

    if True:
        pp.imshow(value.get(), interpolation='nearest')
        pp.colorbar()
        pp.show()

    #mainting copy of target data
    data = ctx.array(value)

    #post-smoothing, to erase superfluous local maxima
    t=clock()
    for i in xrange(10):
        value = averaging(value, stencil = stencil)
    print clock()-t


    #assigns a unique label to each local maximum
    seeder = ctx.kernel(
        """
        (int32[i,j] label = -1, uint32[1] seed = 0) << [i,j] << (<type>[i,j] value):
            value:  padded(stencil)
            label:  padded(stencil)
        {
            uint32 count = 0;
            <type> center = value(i,j);
            for (s in stencil(value))
                 count += center > value(s.i,s.j);
            if (count == stencil.size)
                label(i,j) = atomic_add(seed, 1);
        }
        """)

    label, seeds = seeder(value, stencil = sphere(2,1.9))
    seeds = seeds.get()[0]


    #show label seeds
    if True:
        pp.imshow(label.get(), interpolation='nearest')
        pp.colorbar()
        pp.show()


    #floodfill based watershed kernel
    floodfill = ctx.kernel(
        """
        (<ltype>[i,j] out = in, uint32[1] changed = 0) << [i,j] << (<ltype>[i,j] in, <vtype>[i,j] value):
            in:     padded(stencil)
            out:    padded(stencil)
            value:  padded(stencil)
            i: serial
        {
            <vtype> V = value(i,j);
            <ltype> L = in   (i,j);

            <ltype> old = L;
            float32 s = 0;
            float32 S = 0;

            for (s in stencil(value))
            {
                <vtype> v = value(s.i,s.j);
                <ltype> l = in   (s.i,s.j);
                s = (v - V) / s.weight;

                if (s > S)
                {
                    S = s;
                    L = l;
                }
            }
            if (L != old)
                atomic_add(changed, 1);

            __syncthreads();
            out(i,j) = L;
        }
        """)
    stencil = sphere(2,1.9) #single ring

    t = clock()
    while True:
        label, changed = floodfill(label, value, stencil = stencil)
        print changed
        if changed.get()[0] == 0: break
    print clock()-t

    #compute boundary mask, denoting the places where the label is not constant over the stencil
    boundaries = ctx.kernel("""
        (uint8[n,m] output = 0) << [n,m] << (<type>[n,m] input):
            input:  padded(stencil)
            output: padded(stencil)
            n: serial
        {
            const <type> c = input(n,m);
            uint8 mask = 1;
            for (s in stencil(input))
                 mask = mask && (c == input(s.n,s.m));
            output(n, m) = (0==mask);
        }
        """)
    mask = boundaries(label, stencil = sphere(2,1))

    #find the extents of all unique labels
    find_objects = ctx.kernel("""
        (int32[o,2,2] objects) << [i,j] << (<ltype>[i,j] label):
            o: variable
            i: serial
        {
            <ltype> l = label(i,j);
            if (l!=-1)
            {
                atomic_min(&objects(l,0,0), i);
                atomic_max(&objects(l,0,1), i+1);

                atomic_min(&objects(l,1,0), j);
                atomic_max(&objects(l,1,1), j+1);
            }
        }
        """)
    #initializing objects array is nontrivial, so we manually allocate
    objects = np.zeros((seeds, 2,2), np.int32)
    objects[:,:,0] = 2**24
    objects = ctx.array(objects)
    print find_objects(label, objects)


    measure_objects = ctx.kernel("""
        (float32[o] integral = 0) << [i,j] << (<ltype>[i,j] label, <vtype>[i,j] value):
            o: variable
            i: serial
        {
            <ltype> l = label(i,j);
            <vtype> v = value(i,j);
            if (l!=-1)
            {
                atomic_add(&integral(l), (float32)v);
            }
        }
        """)
    #here the output array is easily allocated behind the scenes
    print measure_objects(label, value, o=seeds)


    #show the end result
    if True:
        pp.figure()
        pp.imshow(np.ma.MaskedArray(data.get(), mask.get()), interpolation='nearest')
        pp.colorbar()
        pp.show()
