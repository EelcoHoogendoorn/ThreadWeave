
"""
improved parsing code

parse global grammar first, then parse subsections
this will keep code more organized
and allows generating more helpful parsing errors


"""


from parsing import *

verbose = False


def structure_grammar():
    """
    global structure of fruityloop incomplete grammar is thus:

        first_line_with_signature_ending_in_colon:
            multi_line_indented_annotations_block
        curly_braced_body
    """

##    signature =  ( SkipTo( Suppress( Literal(':') + LineEnd()), include=True))('signature') + ':'

    signature = SkipTo(Suppress( Literal(':')+LineEnd()), include=True)('signature')


    line = SkipTo(LineEnd())
    stack = [1]
    annotation = originalTextFor(indentedBlock(line, stack))('annotations')
    annotation.addParseAction(lambda s,l,t: dedent(t[0], skip=1))   #skip=1, since first line is already dedented

    body       = originalTextFor(nestedExpr('{','}'))('body')

    return signature + annotation + body
#    return indentedBlock( signature + annotation + body, stack, False)

def structure_parse(source):
    """
    the grammar attempts to split the source string
    into thee components; a signature, an annotation block, and a body
    """
    return structure_grammar().parseString(source)


def signature_grammar():
    """
    build signature syntax
    signature syntax is:
        (outputs) << [kernel] << (inputs)
    or:
        (inputs) >> [kernel] >> (outputs)
    """

    arguments  = originalTextFor(nestedExpr('(', ')')).addParseAction(removeQuotes)
    kernel     = originalTextFor(nestedExpr('[', ']')).addParseAction(removeQuotes)('kernel')

    left  = '<<'
    right = '>>'

    signature_left  = arguments('output') + left  + kernel + left  + arguments('input')
    signature_right = arguments('input')  + right + kernel + right + arguments('output')

    return signature_left | signature_right

def signature_parse(source):
    """
    parses the source string into three components:
        kernel, inputs, and outputs
    """
    return signature_grammar().parseString(source)


def kernel_grammar():
    """
    comma seperated list of axes identifiers, with possible size relations
    """

    identifier = Word(alphas.lower(), exact=1)('identifier')
    cmp_op = Literal('<') | Literal('<=')

    size_indefinite = identifier
    size_equality = identifier + Literal('=') + positive_integer('size')
    size_constraint = Optional( positive_integer('low') + cmp_op) + identifier + Optional( cmp_op  + positive_integer('high'))
    axis = size_equality | size_constraint | size_indefinite

    return delimitedList(Group(axis))

def kernel_parse(source):
    return kernel_grammar().parseString(source)


def input_grammar():
    """
    comma seperated list of input arguments,
    where an input arg has the syntax <type>[shape] identifier
    """
    dummy  = Word(alphas.lower(), exact=1)
    axis   = Or([positive_integer('size'), dummy('identifier')])
    shape  = squareBracketedExpr( ( delimitedList(axis) ) )('shape')
    input  = type_argument + Optional(shape) + identifier

    return Optional(delimitedList(Group(input)))

def input_parse(source):
    return input_grammar().parseString(source)


def output_grammar():
    """
    comma seperated list of input arguments,
    where an input arg has the syntax <type>[shape] identifier
    """
    dummy  = Word(alphas.lower(), exact=1)
    axis   = Or([positive_integer('size'), dummy('identifier')])
    shape  = squareBracketedExpr( ( delimitedList(axis) ) )('shape')
    default = Suppress(Literal('=')) + (number | identifier(''))('default')
    output = type_argument + Optional(shape) + identifier + Optional(default)

    return delimitedList(Group(output))

def output_parse(source):
    return output_grammar().parseString(source)


def annotations_grammar(declaration):
    #kernel axis annotations
    axis_identifier = oneOf(' '.join(declaration.axes_identifiers))('axis')

    parallel = Literal('parallel')
    serial   = Literal('serial')
    hybrid   = Literal('hybrid') + roundBracketedExpr(positive_integer('blocksize'))
    iteration = Or([parallel , serial, hybrid])('iteration')

    variable = Literal('variable')
    template = Literal('template')
    binding = Or([variable, template])('binding')

    axis_keyword = Group( iteration | binding)

    axis_annotation = axis_identifier + Suppress(colon) + OneOrMore( axis_keyword)


    #array annotations
    array_identifier = oneOf(' '.join(declaration.array_identifiers))('array')
    padded  = (Literal('padded') + roundBracketedExpr(identifier('stencil')))
    wrapped = Literal('wrapped')


    boundary = Or([padded , wrapped])('boundary')
    prefetch = Literal('prefetch')

    array_keyword = Group( prefetch | boundary)
    array_annotation =  array_identifier + Suppress(colon) + OneOrMore( array_keyword)


    annotation = Dict(Group(axis_annotation | array_annotation))
    grammar = ZeroOrMore(annotation)
    return grammar

def annotations_parse(declaration, annotations):
    return annotations_grammar(declaration).parseString(annotations)




def input_build(declaration, inputs):
    parsed = input_parse(inputs)
    for i, input in enumerate(parsed):
        input = declaration.input(
            identifier  = input.identifier,
            dtype       = input.dtype,
            shape       = tuple(input.shape))

def output_build(declaration, outputs):
    parsed= output_parse(outputs)
    for i, output in enumerate(parsed):
        output = declaration.output(
            identifier  = output.identifier,
            dtype       = output.dtype,
            shape       = tuple(output.shape),
            default     = output.default)

def kernel_build(declaration, kernel):
    parsed = kernel_parse(kernel)
    for axis in parsed:
        #post-parse inline syntax
        if axis.specified:
            axis = declaration.axis(
                identifier  = axis.identifier,
                size        = axis.size,
                )
        elif axis.bounded:
            axis = declaration.axis(
                identifier  = axis.identifier,
                lower       = axis.lower,
                upper       = axis.upper,
                )
        else:   #unspecified free variable
            axis = declaration.axis(
                identifier  =   axis.identifier,
                )


def signature_build(declaration, signature):
    parsed = signature_parse(signature)

    kernel_build(declaration, parsed.kernel)
    input_build (declaration, parsed.input)
    output_build(declaration, parsed.output)

def body_build(declaration, body):
    """
    scan body for internally declared symbols
    like types and scencils
    should this include splitting the body in pre-mid and post sections?
    or is that better left to body parsing?
    """
    #scan for stencil templates
    stencil_grammar = Literal('for') +roundBracketedExpr(identifier + 'in' + identifier('stencil') + Optional(roundBracketedExpr(identifier)))
    for t,s,e in stencil_grammar.scanString(body):
        declaration.stencil_templates.add(t.stencil)
    #scan for type templates? cant think of use case, and not sure of possible grammar conflicts


def annotations_build(declaration, annotations):
    parsed = annotations_parse(declaration, annotations)
    #parse axis annotations
    for axis in declaration.axes:
        axis_annotations = getattr(parsed, axis.identifier)
        if axis_annotations == '': continue

        for axis_annotation in axis_annotations:
            if axis_annotation.iteration == 'parallel':
                axis.iteration = 'parallel'
            if axis_annotation.iteration == 'serial':
                axis.iteration = 'serial'
            if axis_annotation.iteration == 'hybrid':
                raise NotImplementedError()
            if axis_annotation.binding == 'variable':
                axis.runtime_variable = True
                declaration.size_variables.add(axis.identifier)
    #parse array annotations
    for array in declaration.arrays:
        array_annotations = getattr(parsed, array.identifier)
        if array_annotations == '': continue

        for array_annotation in array_annotations:
            if array_annotation.boundary == 'padded':
                array.boundary = 'padded'
                array.stencil = array_annotation.stencil
                declaration.stencil_templates.add(array.stencil)   #one of those things that should happen at finilization...
            if array_annotation.boundary == 'wrapped':
                array.boundary = 'wrapped'
            if array_annotation.prefetch == 'prefetch':
                array.prefetch = True


def build_declaration(source):

    source = dedent(source)
    if verbose:
        print source

    structure = structure_parse(source)

    #create the declaration
    from ..declaration import KernelDeclaration
    declaration = KernelDeclaration()

    #add information from signature, annotations, and body to declaration
    signature_build  (declaration, structure.signature)
    annotations_build(declaration, structure.annotations)
    body_build       (declaration, structure.body)

    #return the declaration and its body
    return declaration, structure.body







if __name__=='__main__':
    def parse(source):
        """
        general parsing entry point
        prints debug info
        """
        #remove shared indentation
        source = dedent(source)
        print source

        structure = structure_parse(source)

        print structure.signature
        print structure.annotation
        print structure.body


        signature = signature_parse(structure.signature)

        kernel = kernel_parse(signature.kernel)
        inputs = input_parse(signature.input)

        print signature.kernel
        print signature.input
        print signature.output


        quit()

    source = """
        (<type>[i,j] output, uint32 changed) << [i=10,0<j<=3,k] << (<type> input):
            dinges
            nogun dinges2
            i: serial
        {
            float32 r = 0;
            axis(i)
            {
                axis(j)
                {
                    r += 1;
                }
            }
            atomic_add(changed, r);
        }
    """

    ##source = """
    ##sig:
    ##    zzzz
    ##    qwerty
    ##{
    ##}
    ##"""
    r = build_declaration(source)
    print r


