

"""
this module merely defines some general parsing constructs, for use elsewhere

"""



import numpy as np
import string

from pyparsing import *


try:
    import pycuda.tools
    import pycuda.compyte as compyte
except ImportError:
    try:
        import pyopencl.tools
        import pyopencl.compyte as compyte
    except ImportError:
        raise ImportError("no compyte")

dtype_to_ctype = {k:v for k,v in compyte.dtypes.DTYPE_TO_NAME.iteritems() if isinstance(k, str)}
dtypes = dtype_to_ctype.keys()

print 'warning: non-primitive types have a backend-specific name. perhaps its better to let type specifiers be arbitrary identifiers during parsing, and deal with them further during code generation'


#some general grammar definitions
def bracketedExpr(brackets, expr):
    left, right = brackets
    return Suppress(Literal(left)) + expr + Suppress(Literal(right))
def roundBracketedExpr(expr):
    return bracketedExpr('()', expr)
def squareBracketedExpr(expr):
    return bracketedExpr('[]', expr)
def angleBracketedExpr(expr):
    return bracketedExpr('<>', expr)
def curlyBracketedExpr(expr):
    return bracketedExpr('{}', expr)


dtype_term = oneOf(' '.join(dtypes))         #valid numpy-style type string (ie: 'float32')


identifier = Word(alphas+'_', alphanums+'_').setResultsName('identifier')
dummy = Word(alphas.lower(),exact=1)


#numerical defines; probably easy to just use IEEE compatible regexes for this
def sign_wrap(expr): return Combine(Optional(Literal('-')) + expr)
positive_integer        = Word(nums)
integer                 = sign_wrap(positive_integer)
positive_floating       = Combine( Optional( positive_integer) + '.' + Optional(positive_integer))
floating                = sign_wrap(positive_floating)

positive_integer .setParseAction(lambda s,l,t: int(t[0]))
integer          .setParseAction(lambda s,l,t: int(t[0]))
positive_floating.setParseAction(lambda s,l,t: float(t[0]))
floating         .setParseAction(lambda s,l,t: float(t[0]))

number                  = Or([integer, floating])


colon = Literal(':')

type_argument = Or([angleBracketedExpr(identifier), dtype_term])('dtype')



anything = printables+string.whitespace



###simple arithmetic expression grammar
##arithmetic_operator = oneOf('* / + -')
##arithmetic_operand = integer | identifier
##arithmetic_expr = Forward()
##arithmetic_expr << arithmetic_expr | arithmetic_operand
##arithmetic_expr << arithmetic_expr + arithmetic_operator + arithmetic_expr

import os

def indent(source, cols=4):
    return os.linesep.join(' '*cols + line for line in source.splitlines())
##def dedent(self, body):
##    return self.statements(l.strip() for l in body.splitlines())
##    #remove excess whitespace from body
##    lines = body.splitlines()
##    r = min(len(l)-len(l.lstrip()) for l in lines if len(l)>0)
##    return self.statements(l[r:] for l in lines)
def dedent(source, skip = 0):
    """
    dedent a block of lines, to remove shared indentation
    removes whitespace introduced by source specified as multiline string in local indentation
    skip argument allows skipping some lines from this behavior
    to compensate for annoying pyparsing originalTextFor(indentedBlock) behavior
    """
    try:
        lines = source.splitlines()
        indent = min(len(line)-len(line.strip()) for i,line in enumerate(lines) if len(line.strip())>0 and i >= skip)
        return os.linesep.join(line[indent:] if i>= skip else line for i, line in enumerate(lines))
    except:
        return source




