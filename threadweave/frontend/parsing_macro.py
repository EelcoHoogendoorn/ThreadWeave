
"""
macro parsing

this module parses C code with some custom extensions, and upon binding with a code-generator,
it emits valid backend-specific code instead. this serves to paper over the differences
that exist in the backends, and allows the extension of plain C with some preprocessing functionality
that integrate nicely with the rest of fruityloop


the full list of features supported:
    nd-array awareness:
        nd-array indexing (array_id(idx0, idx1...)); round-bracketed to avoid conflict with POC array indexing
        shape attributes on nd-arrays

    stencil support:
        unrolled loops over stencils: ( for (s in stencil) {s.weight;} )
        access to stencil attributes ( stencil.size; )

    fully qualified numpy style typenames (uint32, float64, andsoforth)
    templated type arguments (<type_id>)

    atomic substitution: (openCL syntax is used)

    integer looping ( for (id in start:stop){} )

many more features could easily be added, as long as they are fully backwards compatible with POC

"""

from parsing import *



def replace_atomics(cg, source):
    """
    replace openCL-style atomics with backend specific atomic operations
    """
    replacement = cg.atomic_op

    op = oneOf('add sub inc dec min max')
    atomic_grammar = Literal('atomic_') + op('op')

    def replace(s,l,t):
        return replacement(t.op)
    atomic_grammar.setParseAction(replace)
    return atomic_grammar.transformString(source)


def replace_type_templates(cg, source):
    """
    replace numpy types with c-types. this could be more efficient and intelligent...
    we do not do any semantic analysis here; simple find and replace
    but that should suffice, no?
    """
    type_grammar = angleBracketedExpr(identifier('template'))
    def replace(s,l,t):
        return cg.declaration.templates[t.template]
    type_grammar.setParseAction(replace )
    return type_grammar.transformString(source)


def replace_typing(cg, source):
    """
    replace numpy types with c-types. this could be more efficient and intelligent...
    we do not do any semantic analysis here; simple find and replace
    but that should suffice, no?
    """
    replacement = cg.type_specifier
    type_grammar = dtype_term.copy()
    type_grammar.setParseAction(lambda s,l,t: replacement(t[0]))
    return type_grammar.transformString(source)


def replace_shape_syntax(cg, source):
    """
    replace arrayidentifier.shape[ndim] syntax with C named variables
    silently fails to replace some wrong syntax, like misspelled shape;
    dont worry, the cuda compiler is sure to complain about it :)
    would it be sufficient and currect to catch all instances of 'arrayidentifier.'+whatever,
    that fail to match the whole syntax?
    """
    declaration = cg.declaration

    arrayidentifier = (Word(alphanums+'_')).setResultsName('identifier') # + Optional( Word(alphanums))
    positive_integer = Word(nums)
    shape_expr = arrayidentifier + Suppress( Literal('.shape')) + nestedExpr('[',']', positive_integer).setResultsName('dimension')

    def replace(s,l,t):
        """if match is correct, replace numpy syntax with c-compatible syntax"""
        identifier = t.identifier
        dimensions = t.dimension[0]
        if not len(dimensions)==1: raise Exception('only simple shape indexing allows')
        dimension = dimensions[0]
        try:
            arg = declaration.arguments[identifier]
        except KeyError:
            raise ParseFatalException("array '{identifier}' is not defined".format(identifier=identifier))
        try:
            size = arg.shape[int(dimension)]
        except Exception:
            raise ParseFatalException('{identifier}.shape[{dimension}] is invalid'.format(identifier=identifier, dimension=dimension))

        return '{identifier}_shape_{dimension}'.format(identifier=identifier, dimension=dimension)
    shape_expr.setParseAction(replace)

    return shape_expr.transformString(source)


def replace_array_syntax(cg, source):
    """
    replace weave.blitz style array indexing with inner product over strides
    we could optionally insert bounds checking code here as well, as a debugging aid
    should we allow for partial indexing? not sure; disallowed atm

    we could do compile-time bounds checking, for cases where the shapes are know at compile time
    and the presumably predictable reserved indexing symbols i{n} are used
    """
    declaration = cg.declaration
    replacement = cg.indexing_expression

    arrayidentifier = oneOf(' '.join(declaration.array_identifiers))('arrayidentifier')
##    index = Or([identifier, positive_integer])
##    index = Word( printables.replace('()',''))
    #note; updated parsing expression allows for braced expressions within the indexing statement
    index_expr = arrayidentifier + originalTextFor( nestedExpr())('indices')

    def replace(s,l,t):
        """if match is correct, replace numpy syntax with c-compatible syntax"""
        identifier = str( t.arrayidentifier)
        indices = tuple(t.indices.strip('()').split(','))

        try:
            arg = declaration.arguments[identifier]
        except KeyError:
            raise ParseFatalException("array '{identifier}' is not defined".format(identifier=identifier))

        if not len(indices)==arg.ndim:
            raise Exception("indexing '{identifier}' requires {ndim} arguments".format(identifier=identifier, ndim=arg.ndim))

        return replacement(arg=arg, indices=indices)

    index_expr.setParseAction(replace)
    return index_expr.transformString(source)



def replace_for_syntax(cg, source):
    """
    replace: 'for (id in start:stop:step)'
    with:    'for (int id=start; start<stop; id+=step)'
    rather trivial syntactic sugar indeed
    we could implement an unrolling mechanism here too,
    in case all params are known at compile time, and loop is small?
    """
    replacement = cg.for_statement

    index = Or([sign_wrap(identifier), integer])
    colon = Suppress(Literal(':'))
    range = index.setResultsName('start') + colon + index.setResultsName('stop') + Optional(Combine(colon + index), '1').setResultsName('step')
    loop_expr = Literal('for') + '(' + identifier.setResultsName('index') + Literal('in') + range + ')'

    loop_expr.setParseAction(lambda s,l,t: replacement(**dict(t)))
    return loop_expr.transformString(source)


def replace_for_stencil_body(dummy, body):
    """
    expand stencil_dummy_id.property syntax
    to C-compatible underscore names
    """
    grammar = Literal(dummy) + Literal('.').setParseAction(replaceWith('_'))
    return grammar.transformString(body)

def replace_stencil_properties(cg, body):
    """
    expand stencil_dummy_id.property syntax
    to C-compatible underscore names
    """
    declaration = cg.declaration

    stencil_identifier = oneOf(' '.join(declaration.stencil_templates))
    attribute = Literal('size')('attribute')
    grammar = stencil_identifier('identifier') + '.' + attribute
    def replace(s,l,t):
        if t.attribute=='size':
            return declaration.get_stencil(t.identifier).size
    grammar.setParseAction(replace)
    return grammar.transformString(body)


def replace_for_stencil_syntax(cg, source):
    """
    replace: 'for (id in stencilid)'
    with:    iteration over the body
    expand dummy accessors as well, like s.weight, and s.kernel_axis
    this parsing operation needs to be nested
    first we must parse the declaration to obtain the dummy identifier
    only then can we cleanly parse the body

    only do unrolled loop for now
    giver each term in loop its own scope, so we can redeclare const vars within?

    need to do some thinking; what does the stencil loop over? an array or the kernel?
    we do padding per array; this argues for stenceling per array as well
    that is, we need to decide on a set of axes to apply the stencil over
    or should we simply generate a tuple of nameless offsets, and let
    the end user handle the indexing?
    could use array indexing syntax to get offsets
    or, explicitly declare the axes to broadcast the stencil over
    as in, for (s in stencil(i,j,k)) { array(s.i), andsoforth}
    nah; just broadcast stencil to match array dimension
    then bind to the array, copying the axis names of the array bound
    """

    declaration = cg.declaration

    unroll = True #deduce from syntax? or stencil object?

    array = oneOf(' '.join(declaration.array_identifiers))
    for_range = identifier('dummy') + 'in' + identifier('stencil') + roundBracketedExpr(array('array'))
    scoped_body = Combine( nestedExpr('{','}', Word(anything.translate(None, '{}'))))
    scoped_body.setParseAction(keepOriginalText)
    line_body = Combine( SkipTo(';', include=True))
    body = Or([scoped_body, line_body])('body')
    stencil_expr = (Literal('for') + roundBracketedExpr(for_range) + body)


    def replace(s,l,t):
        body = t.body
        dummy = t.dummy

        array = declaration.arguments[t.array]

        replaced_body = replace_for_stencil_body(dummy, body)


        def voxel(v):
            #get all stencil props here; list of tuples of offsets?
            #ideally, pull such constructs from codegen

            index   = 'const int {d_id}_{axis_id} = (int){axis_id} + {offset};'
            indexes = cg.statements(
                [index.format(offset=o, axis_id=a.identifier, d_id = dummy)
                    for o, a in zip(v.offset, array.shape)], newline=False)

            weight = 'const float32 {d_id}_weight = {weight};'.format(weight=v.weight, d_id=dummy)

            scope = cg.scope(
                cg.statements([
                    indexes,
                    weight,
                    replaced_body]))
            return scope


        #grab stencil from the template dict
        #if not implied by a padding keyword, explicitly add stencil decl to signature?
        stencil = declaration.get_stencil(t.stencil)

        return ''.join(voxel(v) for v in stencil.voxels())


    stencil_expr.setParseAction(replace)
    return stencil_expr.transformString(source)


def replace_output_reference_syntax(source):
    """
    scalar output args are passed by reference; they are actually shape=(1,) arrays
    we might need them by reference for atomic operations
    perhaps just add a & in those cases; treat it as a value type by placing a * everywhere
    in the body of code. does cuda C handle &*ptr==ptr correctly?
    """


def parse_body(body):
    """
    lazily parse body code
    that is, we return a function that will perform the parsing upon recieving a matching code generator
    """

    def inner(code_generator):
        b = dedent( body)

        b = replace_stencil_properties  (code_generator, b)
        b = replace_for_syntax          (code_generator, b)
        b = replace_for_stencil_syntax  (code_generator, b)
        b = replace_type_templates      (code_generator, b)
        b = replace_typing              (code_generator, b)
        b = replace_array_syntax        (code_generator, b)
        b = replace_atomics             (code_generator, b)
        return b

    return inner

