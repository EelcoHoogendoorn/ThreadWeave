

"""
parsing module
exports functions to transform source code into a kernel declaration object

this is where the grammar is defined
out arbitrary
should we allow expressions as size constraints?
usecase; subsampling an image, f.i. or differentiating a vector
cute feature, but not crucial
simply store these constraints as strings, and eval them,
after shape arguments have been harvested
or use sympy to reason about arbitrary constraint set


this module merely defines some general parsing constructs
"""



import numpy as np
import string

from pyparsing import *

import pycuda.tools
dtype_to_ctype = {k:v for k,v in pycuda.compyte.dtypes.DTYPE_TO_NAME.iteritems() if isinstance(k, str)}

dtypes = dtype_to_ctype.keys()


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


#numerical defines
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




