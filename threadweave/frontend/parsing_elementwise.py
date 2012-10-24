
"""
parses an elementwise into a kernel factory

"""

from parsing import *



def elementwise_grammar():
    dtype = oneOf('float32 float64 int32 int64 uint32 uint64')

    grammar = Optional( dtype('dtype') + colon) + SkipTo( LineEnd())('expression')

    return grammar


def expression_check(expression):
    """
    check if dummies are monotonically increasing, and return count
    """
    dummies = set()
    def action(s,l,t):
        dummies.add(int(t[0]))
    dummy = curlyBracketedExpr(positive_integer).setParseAction(action)
    for q in dummy.scanString(expression):
        pass
    count = len(dummies)
    assert(set(xrange(count)) == dummies)
    return count

def expression_transform(expression):
    """
    transform expression into valid c code, with a substitution slot for insertion of arbitrary number of axes
    """
    def action(s,l,t):
        i = str(t[0])
        return 'input_'+i+'({idx['+i+']})'

    dummy = curlyBracketedExpr(positive_integer).setParseAction(action)
    return dummy.transformString(expression)


def parse(source):
    parsed = elementwise_grammar().parseString(source)
    print parsed.dtype
    print parsed.expression

    arguments = expression_check(parsed.expression)
    return arguments, expression_transform(parsed.expression)


if __name__=='__main__':
    source = "float32: {0} + cos({1})"
    parse(source)