
"""
tensor product declaration front end language definition

syntax similar to einsum, but with seperate compilation and invocation stage

big question; how to implement contractions sanely? loopy to the rescue here?
for now, we have a reference implemntations that relies on atomics (ouch)

some ideas that suck a little less:
    looping axes that are reducing can work with a single register;
    inited at start of serial axis, and presented as output at the end; think this through!
    all parallel axes in a threadblock sum to a single value in shared mem
    this is then written to global mem using an atomic, at the end

we need some extra syntax to be able to tweak parallel/serial axes
could use lower/upper case index? or additional terms beyond initial string.
do not want to dillute tensor product itself, in any case

try writing simple example code for parallel and serial reduction axis first
we need arbitrary chainings though; take tensor prod 'ijk->j' f.i.
"""

"""
serial reduction over axis i:
    <type> kernel_i = 0;
    for (i=0; i<i_size;i++)
    {
        kernel := tensor_product_here
        kernel_i {reductionop} kernel
    }
    output(idx) = kernel_i
"""

"""
parallel reduction:
    //sum block in shared memory
    //block-reduced value is atomic-added?
"""
from parsing import *




def parse(source):
    """parse einsum-like source string to declaration datastructure"""

    axes = set()    #mine all axes used in the declaration

    index = Word(alphas.lower(), exact=1).setParseAction(lambda s,l,t: axes.add(str(t[0])))
    tensor = Group( OneOrMore(index))
    arrow = Literal('->')
    syntax = delimitedList(tensor)('inputs') + arrow + tensor('output')

    parsed = syntax.parseString(source)

    axes = sorted(axes)

    #build a declaration
    from .. import declaration
    decl = declaration.KernelDeclaration()

    #add axes to declaration
    contraction = False
    for axis in axes:
        #a decent implementation would not only notice reduction axes
        #but actually use that information in code-gen too
        contracted = not axis in list(parsed.output)
        axis = decl.axis(identifier=axis, reduction = contracted)
        if contracted:
            contraction = True
            axis.iteration = 'serial'

    #add inputs
    for i, axes in enumerate(parsed.inputs):
        input = decl.input(identifier = 'input_{i}'.format(i=i), dtype='type', shape = tuple(axes))

    #add output
    output = decl.output(identifier = 'output', dtype='type', shape = tuple(parsed.output), default = 0)
    assert(not any(output.shape.count(axis) > 1 for axis in output.shape))      #repeated output axes are disallowed

    #construct body in terms of macro language, to handle array indexing
    #would be cleaner to have and use a body declaration API here...
    def axes_idx(arg):
        return ','.join(axis.identifier for axis in arg.shape)
    input = '*'.join('input_{i}({idx})'.format(i=i, idx=axes_idx(input)) for i,input in enumerate( decl.inputs))
    if contraction:
        body = "atomic_add( &output({idx}), {input});".format(idx = axes_idx(output), input = input)
    else:
        body = "output({idx}) = {input};".format(idx = axes_idx(output), input = input)


    from parsing_macro import parse_body
    decl.body = parse_body(body)

    return decl


if __name__ is '__main__':
    #some unit tests
##    parse('ij,k->ijk')

    print parse('ij,jk->ik')
