"""Retrieving the object and corresponding source blocks at the same
time."""

import preamble

import ast, inspect, textwrap
from pprint import pprint

from collections import OrderedDict

def grammar(func):
    src = inspect.getsource(func)
    src_lns = src.splitlines()
    md = ast.parse(src)
    fd = md.body[0]
    objs = fd.body

    src_blks = []
    decls = []
    ctx = {}
    for c, c1 in zip(objs, objs[1:]):
        i, j = c.lineno - 1, c1.lineno - 1
        bsrc = textwrap.dedent('\n'.join(src_lns[i:j]))
        src_blks.append(bsrc)
        exec(bsrc, func.__globals__, ctx)
        decls.append(ctx.popitem())

    return decls, src_blks


@grammar
def Calc():

    IGNORED = r' '
    IGNORED = r'\t'

    EQ  = r'='
    NUM = r'[0-9]+'
    ID  = r'[_a-zA-Z]\w*'
    POW = r'\*\*', 3
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    def assign(ID, EQ, expr):
        table[ID] = expr
        return expr

    def expr(NUM):
        return int(NUM)

    def expr(expr_1, ADD, expr_2):
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1


pprint(Calc)
