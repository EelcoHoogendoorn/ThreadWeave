"""
some unit tests i made to test if my code edits break any functionality
kindof a mess, but has some tutorial like utility
the good bits from it should be put in the examples dir, with good commentary
"""

import matplotlib.pyplot as pp
import numpy as np
from time import clock

#we can make a drop-in replacement of the backend, and everything should work the same
#this is the only place where end-user code will be forced to see anything backend-specific
#each context aims to expose an interface that resembles the numpy namespace (only bare minimum implemented!)
#on top of that, one is free to use backend specific features through this context object, of course
#if cross-backend compatibility is of no concern
if True:
    from ..backend import CUDA as Backend
else:
    from ..backend import OpenCL as Backend

device = 0



def test_convolution():


    #showcase convolution functionality
    #convolution kernels could be added to the toolbox of the context
    #then again; standard convolution kernels are already part of many libs;
    #the added fun eith fruityloop is in the customizations
    #such as boundary conditions, axis types, and dimensionality
    #only the body code has any degree of generality; but it is only 4 lines

    from ..stencil import laplacian

    with Backend.Context(device) as ctx:

        #declare the laplacian kernel
        #this kernel is intended for stacks of 2d arrays. this could be done by means of repeasted calls
        #to a 2d stencil, but in doing so we can expose additional parallelism
        #n,m are compile time constants which are JITed into the kernel
        #d is a runtime variable, so that repeated calls on varying stack depths do not cause recompilation
        #(this is just an example; alternate configurations may be more appropriate depending on the expected input pattern)
        #boundary conditions are handled by means of padding. the padding is chosen to accomodate the stencil used
        #this padding is taken to mean that the kernel only acts on a safe view of the supplied data

        laplacian_kernel = ctx.kernel(
            """
            (<type>[d,n,m] output = 0) << [d,n,m] << (<type>[d,n,m] input):
                n:      serial
                d:      variable
                input:  padded(stencil)
            {
                <type> r = 0;
                for (s in stencil(input))
                     r += input(s.d,s.n,s.m) * s.weight;
                output(d,n,m) = r;
            }
            """)

        #define a simple input array; a quadratic in all directions
        #the laplace of this quadratic function should be twice the sum of squared strides
        shape = (4,6,6)
        input = ctx.array(np.arange(np.prod(shape)).astype(np.float32).reshape(shape) ** 2)
        print input

        #we use a 2-d 5-point laplacian, broadcasted over the stacking dimension
        stencil = laplacian(2,5)[np.newaxis,:,:]
        #invoke the kernel with the given input and stencil.
        #the output array shape is deduced and allocated behind the scenes
        print laplacian_kernel(input, stencil=stencil)


def test_segmentation():

    #hybrid watershed/threshold segmentation
    #rather, just a watershed, acting on a smoothed variant

    from ..stencil import sphere

    with Backend.Context(device) as ctx:

        #averaging filter
        #init output with copy of input
        #this ensures copying over the padding, in case of padded arrays
        #wrapped access is implemented as integer modulus
        #this only works if the array shape is a power of two, in which case
        #CUDA
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
        stencil = sphere(2,1.5,0)

        #construct smoothed random data
        value = ctx.random((256,256), np.float32)
        for i in xrange(5):
            value = averaging(value, stencil = stencil)

        #do some transforms on data, to make it more interesting for segmentation
        value -= value.get().min()
        value /= value.get().max()
        value **= 3

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


        if False:
            pp.imshow(label.get(), interpolation='nearest')
            pp.colorbar()
            pp.show()


        floodfill = ctx.kernel(
            """
            (<ltype>[i,j] out, int32[1] changed = 0) << [i,j] << (<ltype>[i,j] in, <vtype>[i,j] value):
                in:     padded(stencil)
                out:    padded(stencil)
                value:  padded(stencil)

                i: serial
            {
                //allocate and init associative array
                int32   labels   [stencil.size];
                float32 capitance[stencil.size];
                for (q in 0:stencil.size)
                {
                    labels[q] = -1;
                    capitance[q] = 0;
                }

                <vtype> V = value(i,j);
                <ltype> L = in   (i,j);

                <ltype> old = L;

                if (1)
                {
                    for (s in stencil(value))
                    {
                        <vtype> v = value(s.i,s.j);
                        <ltype> l = in   (s.i,s.j);

                        if (l!=-1 && v > V)
                        {
                            //find index in associative array
                            int32 q = 0;
                            while(1)
                            {
                                if (labels[q]==l) break;
                                if (labels[q]==-1){labels[q]=l; break;}
                                q++;
                            }
                            capitance[q] += (v - V) / s.weight;
                        }
                    }

                    //find label of max capitance
                    float32 C = 0;
                    for (q in 0:stencil.size)
                    {
                        float32 c = capitance[q];
                        if (c > C)
                        {
                            C = c;
                            int32 l = labels[q];
                            if (l==-1) break;
                            L = l;
                        }
                    }
                }
                else
                {
                    for (s in stencil(value))
                    {
                        <ltype> l = in   (s.i,s.j);
                        L = max(l,L);
                    }
                }

                //check if this leads to a changed label
                if (L != old)
                    atomic_add(changed, 1);

                __syncthreads();    //make sure we all write at the same time
                out(i,j) = L;       //write in any case, since output is not initialized
            }
            """)
        floodfill = ctx.kernel(
            """
            (<ltype>[i,j] out, int32[1] changed = 0) << [i,j] << (<ltype>[i,j] in, <vtype>[i,j] value):
                in:     padded(stencil)
                out:    padded(stencil)
                value:  padded(stencil)

                i: serial
            {
                //allocate and init associative array
                int32   labels   [stencil.size];
                float32 capitance[stencil.size];
                for (q in 0:stencil.size)
                {
                    labels[q] = -1;
                    capitance[q] = 0;
                }

                <vtype> V = value(i,j);
                <ltype> L = in   (i,j);

                <ltype> old = L;

                for (s in stencil(value))
                {
                    <vtype> v = value(s.i,s.j);
                    <ltype> l = in   (s.i,s.j);

                    if (l!=-1 && v > V)
                    {
                        //find index in associative array
                        int32 q = 0;
                        while(1)
                        {
                            if (labels[q]==l) break;
                            if (labels[q]==-1){labels[q]=l; break;}
                            q++;
                        }
                        capitance[q] += (v - V) / s.weight;
                    }
                }

                //find label of max capitance
                float32 C = 0;
                for (q in 0:stencil.size)
                {
                    float32 c = capitance[q];
                    if (c > C)
                    {
                        C = c;
                        int32 l = labels[q];
                        if (l==-1) break;
                        L = l;
                    }
                }

                //check if this leads to a changed label
                if (L != old)
                    atomic_add(changed, 1);

                __syncthreads();    //make sure we all write at the same time
                out(i,j) = L;       //write in any case, since output is not initialized
            }
            """)

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

                __syncthreads();    //make sure we all write at the same time
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
        if False:
            #disabled, since this test fails for openCL
            print measure_objects(label, value, o=seeds)



        if True:
            pp.figure()
            pp.imshow(np.ma.MaskedArray(data.get(), mask.get()), interpolation='nearest')
            pp.colorbar()
            pp.show()


##def test_stencil():
##
##    with Backend.Context(device) as ctx:
##
##        from ..stencil import laplace_2_5, sphere
##
##        #test stencil functionality
##        #stencil functionality consists of several parts
##        #a stencil kernel requires a stencil object according to the interface defined in the
##        #stencil submodule, where some common stencils are predefined
##        #in code, we can loop over the voxel in a stencil with the syntax:
##        #for (voxel_id in stencil_id(array){}
##        #this binds the stencil to the specified array, and gives access
##        #to all the voxels properties with the syntax voxel_id.property
##        #we can thus access the weight of the voxel, and offsetted indices of the array we bound to
##        #stencil operations require a method to handle the boundary of arrays
##        #to this end, array arguments can be annotated with a boundary handling mode keyword
##        #the input array is padded in this case, which amounts to saying input is to be treated as
##        # a view on a larger block of memory. on the padding outside the view, we can then impose
##        #arbitrary boundary conditions before making the call
##        #this is a very general way of specifiying boundary conditions, which has no overhead inside the kernel
##        #other boundary handling modes that are easily implemented are wrapping and clamping
##        #though these might have significant performance implications.
##        #clamping requires two branches per axes
##        #wrapping can be implemnted with two branches per axis as well
##        #for axes with a size of powers of two, a bitmask can be used
##        #this is the only option currently implemented
##
##
##        averaging = ctx.kernel(
##            """
##            (<type>[n,m] output = 0) << [n,m] << (<type>[n,m] input):
##                input:  padded(stencil)
##                output: padded(stencil)
##            {
##                <type> r = 0;
##                for (s in stencil(input))
##                     r += input(s.n,s.m);
##                output(n, m) = r / stencil.size;
##            }
##            """)
##        value = ctx.random((256+2,256+2), np.float32)
##        T = {'stencil':sphere(2,1.5,0)}
##        for i in xrange(200):
##            value = averaging(value, templates = T)
##
##
##        #assigns a unique label to each local maximum
##        seeder = ctx.kernel(
##            """
##            (int32[i,j] label = -1, uint32[1] seed = 0) << [i,j] << (<type>[i,j] value, <type> treshold):
##                value:  padded(stencil)
##                label:  padded(stencil)
##            {
##                uint32 count = 0;
##                <type> center = value(i,j);
##                if (center > treshold)
##                {
##                    for (s in stencil(value))
##                         count += center > value(s.i,s.j);
##                    if (count == stencil.size)
##                        label(i,j) = atomic_add(seed, 1);
##                }
##            }
##            """)
##
##        label, seeds = seeder(value, np.float32(0.0), templates = {'stencil':sphere(2,1.9)})
##
##
##        #naive floodfill; simply spills each pixel to the steepest neighbor in the stencil
##        #this can lead to surprisingly disappointing results
##        floodfill = ctx.kernel(
##            """
##            (<ltype>[i,j] out, uint32[1] changed = 0) << [i,j] << (<ltype>[i,j] in, <vtype>[i,j] value):
##                in:     padded(stencil)
##                out:    padded(stencil)
##                value:  padded(stencil)
##            {
##                <vtype> V = value(i,j);
##                <ltype> L = in   (i,j);
##
##                <ltype> old = L;
##                float32 s = 0;
##                float32 S = 0;
##
##                for (s in stencil(value))
##                {
##                    <vtype> v = value(s.i,s.j);
##                    <ltype> l = in   (s.i,s.j);
##                    s = (v - V) / s.weight;
##
##                    if (s > S)
##                    {
##                        S = s;
##                        L = l;
##                    }
##                }
##                if (L != old)
##                    atomic_add(changed, 1);
##
##                __syncthreads();    //make sure we all write at the same time
##                out(i,j) = L;
##            }
##            """)
##        #better floodfill. this one scans the whole stencil, and computes the label of least resistance by means of an associative array
##        #this counteracts forming of rather a-physical thin drainage paths
##        floodfill = ctx.kernel(
##            """
##            (<ltype>[i,j] out, uint32[1] changed = 0) << [i,j] << (<ltype>[i,j] in, <vtype>[i,j] value):
##                in:     padded(stencil)
##                out:    padded(stencil)
##                value:  padded(stencil)
##            {
##                //allocate and init associative array
##                int32   labels   [stencil.size];
##                float32 capitance[stencil.size];
##                for (q in 0:stencil.size)
##                {
##                    labels[q] = -1;
##                    capitance[q] = 0;
##                }
##
##                <vtype> V = value(i,j);
##                <ltype> L = in   (i,j);
##
##                <ltype> old = L;
##
##                for (s in stencil(value))
##                {
##                    <vtype> v = value(s.i,s.j);
##                    <ltype> l = in   (s.i,s.j);
##
##                    if (l!=-1 && v > V)
##                    {
##                        //find index in associative array
##                        int32 q = 0;
##                        while(1)
##                        {
##                            if (labels[q]==l) break;
##                            if (labels[q]==-1){labels[q]=l; break;}
##                            q++;
##                        }
##                        capitance[q] += (v - V) / s.weight;
##                    }
##                }
##
##                //find label of max capitance
##                float32 C = 1e-20;
##                for (q in 0:stencil.size)
##                {
##                    float32 c = capitance[q];
##                    if (c > C)
##                    {
##                        C = c;
##                        int32 l = labels[q];
##                        if (l==-1) break;
##                        L = l;
##                    }
##                }
##
##                //check if this leads to a changed label
##                if (L != old)
##                    atomic_add(changed, 1);
##
##                __syncthreads();    //make sure we all write at the same time
##                out(i,j) = L;       //write in any case, since output is not initialized
##            }
##            """)
##        T = {'stencil':sphere(2,1.9)}   #single ring
##        while True:
##            label, changed = floodfill(label, value, templates = T)
##            print changed
##            if changed.get()[0] == 0:
##                break
##
##        print 'test speed of local mem use'
##        print 'create syntax for thread local memory? as in; a chunk of shared for every thread'
##        print 'syntax for associative array?'
##
##
##        if True:
##            import matplotlib.pyplot as pp
##            pp.figure()
##            pp.imshow(label.get(), interpolation='nearest')
##            pp.colorbar()
##            pp.figure()
##            pp.imshow(value.get(), interpolation='nearest')
##            pp.colorbar()
##            pp.show()
##
##
##
##        #finds the min/max of each object in the labelled image
##        #not super efficient, but not the bottleneck in this algo anyway
##        find_objects = ctx.kernel("""
##        (uint32[o,2,2] objects) << [i,j] << (<ltype>[i,j] label):
##            label:  padded(stencil)
##        {
##            <ltype> l = label(i,j);
##
##            atomic_min(&objects(l, 0, 0), i);
##            atomic_max(&objects(l, 0, 1), i+1);
##
##            atomic_min(&objects(l, 1, 0), j);
##            atomic_max(&objects(l, 1, 1), j+1);
##        }
##        """)
##        #object array is output arg, but allocation and initialization is nontrivial
##        #therefore, we do so manually, and pass in the output arg into the kernel call
##        #outputs are to be supplied after the (required) inputs; this overrides internal allocation
##        #also, the properties of the output array are now available for template argument inferrence
##        seeds = seeds.get()[0]
##        obj = np.zeros((seeds, 2,2),np.uint32)
##        obj[...,0] = 2**25
##        obj = ctx.array(obj)
##
##        obj = find_objects(label, obj, templates = T)
##        print obj
##
##
##        print






def test_utility():
    with Backend.Context(device) as ctx:
        #2d meshgrid function;
        #showcases some basic features
        #the high level syntax is
        #(input_list) << [kernel_Axes] << (outputs_list):
        #   annotations
        #{body}
        mesh_grid_2 = ctx.kernel(
            """
            (<type>[2,n,m] result) << [n,m] << ():
                n: serial
            {
                result(0,n,m) = n;
                result(1,n,m) = m;
            }
            """,
        )

        #the function as declared has no input arguments, but does have three unbound parameters
        #calling with the unbound parameters will allocate the output argument accordingly,
        #execute the kernel, and return the desired output array
        print mesh_grid_2(n=5,m=4, type='int32')


def test_elementwise():
    """
    this test showcases the threadweave elementwise capability
    by building on the nd-awareness of threadweave, the elementwise operations
    in threadweave can easily be extended to allow for broadcasting operation
    (and operations between arbitrarily strided arrays, once this is supported at the ndarray level)

    TODO: overload operators in ndarray class
    """
    with Backend.Context(device) as ctx:
        #allocate some simple test arrays
        a = ctx.arange(4,  dtype=np.float32).reshape((4,1))
        b = ctx.arange(4,  dtype=np.float32).reshape((1,4))
        c = ctx.arange(16, dtype=np.float32).reshape((4,4))

        #elementwise kernels can be defines using these simple expressions
        #involving curly-braced argument positions
        broadcasting_product = ctx.elementwise_kernel('{0}*{1}')
        print broadcasting_product(a,b)

        #there is derived helper functionality for binary operations, specifically
        broadcasting_sum = ctx.binary_elementwise_kernel('+')
        print broadcasting_sum(a,b)

        #but we can also take it in the other direction, of more compicated expressions
        funky_expression = ctx.elementwise_kernel('{0}*{1}+{2}')
        print funky_expression(a, b, c)





def test_products():
    #three different outer products on vectors

    with Backend.Context(device) as ctx:

        #the first example showcases some rich kernel and array axis declaration syntax
        #the core signature information is arranged on the first line
        #and a second optional annotation section contains additional array and axis properties
        #we can explicitly specify if an axis is looped over serially or in parallel
        #the variable keyword forces an axis to be a runtime variable;
        #generally we want to JIT-bake in as much info as possible,
        #but if n can take on many different sizes, the compilation overhead could become prohibitive
        outer = ctx.kernel(
            """
            (<type>[n,m] result = 0) << [n,m] << (<type>[n] x0, <type>[m] x1):
                n:  serial variable
                m:  parallel
            {
                result(n, m) = x0(n) * x1(m);
            }
            """)
        print outer(ctx.arange(4, np.float32), ctx.arange(5, np.float32))


        #this accomplishes the same thing as above
        #however, this subproblem (tensor multiplication) allows for a much simplified syntax,
        #the body is implied by the context, and no identifiers need be known
        #however, the same declaration API, backends and runtime logic is used
        outer = ctx.prod("""i,j->ij""")
        print outer(ctx.arange(4, np.float32), ctx.arange(5, np.float32))

        #numpy reference result
        outer = np.outer
        print outer(np.arange(4), np.arange(5))


        #here we declare another tensor product, but this time with contraction
        #j is recoginzed and handled as a summing reduction axis
        #but while it works, this code is not very efficient at present,
        #since it performs a naive contraction using atomics
        #therefore, it will also fail under openCL; loopy to the rescue?
        matrix_product = ctx.prod("""ij,jk->ik""")

        #need to create clean axis annotation syntax for prod interface as well
        #otoh; demonstration of manually tweaking a declaration (axes should be accessible by identifier...)
        matrix_product.declaration.axes[1].parallel = False

        shape = 3,4
        A = ctx.arange(np.prod(shape),np.int32).reshape(shape)
        shape = 4,5
        B = ctx.arange(np.prod(shape),np.int32).reshape(shape)
        C = matrix_product(A, B)
        print C

        #for comparison, the numpy result
        matrix_product = np.dot
        shape = 3,4
        A = np.arange(np.prod(shape),dtype=np.int32).reshape(shape)
        shape = 4,5
        B = np.arange(np.prod(shape),dtype=np.int32).reshape(shape)
        C = matrix_product(A, B)
        print C


test_utility()
quit()
test_convolution()
test_segmentation()
test_products()