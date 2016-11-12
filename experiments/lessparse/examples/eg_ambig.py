import preamble
from metaparse import *


class pExpr(metaclass=GLR.meta):

    'An ambigious grammar for arithmetic expressions.'

    def plus(lex: r'\+'):
        return lex

    def times(lex: r'\*'):
        return lex

    def number(lex: r'\d+'):
        return int(lex)


    def expr(expr, plus, expr_1):
        return expr + expr_1

    def expr(expr, times, expr_1):
        return expr * expr_1

    def expr(number):
        return number


inp = '2 + 1 * 3'

tks = list(pExpr.lexer.tokenize(inp))

from pprint import pprint

pprint(tks)

r = pExpr.prepare_generalized()
next(r)

for tk in tks:
    x = r.send(tk)
    pprint(x)
else:
    x = r.send(END_TOKEN)
    pprint(x)

# Keep sending further tokens!
tks = list(pExpr.lexer.tokenize(' +  + 1'))
for tk in tks:
    rs = r.send(tk)
    for e in rs:
        if isinstance(e, ParseError):
            pprint(e.args)
        else: break
    else:
        pprint(rs)
else:
    x = r.send(END_TOKEN)
    pprint(x)

