import re

from collections import OrderedDict
from collections import namedtuple

_LEX = OrderedDict(
    IGN = r'[ \t\n]+',
    Lp  = r'\(',
    Rp  = r'\)',
    Lb  = r'\[',
    Rb  = r'\]',
    LB  = r'\{',
    RB  = r'\}',
    LAM = r'\(lambda',
    NUM = r'[+-]?([1-9]\d*|0)(\.\d*)?([Ee][+-]?\d+)',
    ID  = r'[^\[\]\(\)\{\} ]+'
)

LEX = OrderedDict(
    (lex, re.compile(pat))
    for lex, pat in _LEX.items())

def tokenize(inp):
    i = 0
    while i < len(inp):
        for lex, pat in LEX.items():
            m = pat.match(inp, pos=i)
            if m:
                if lex != 'IGN':
                    yield lex, m.group()
                _, i = m.span()
                break
        else:
            print('Unrecognized character {} at inp[{}]'.format(inp[i], i))
            i += 1
    yield None, None

inp = """
(lambda (n) (if (< n 2) 1.0 2.0))
"""

list(tokenize(inp))

import ast

Tree = namedtuple('Tree', 'p cs')

LSEP = 'Lp Lb LB'.split()
RSEP = 'Rp Rb RB'.split()

def parse_par(tkr):
    pars = []
    while 1:
        lex, val = next(tkr)
        if lex not in RSEP:
            pars.append(val)
        else:
            break
    return pars

def parse_lam(tkr):
    pars = parse_par(tkr)
    bdy = parse_one(tkr)
    return ('lambda', pars, bdy)
    # args = ast.arguments(args=[
    #     ast.arg(arg=p, annotation='') for p in _pars])
    # body = _bdy
    # return ast.FunctionDef('', [], [],)

def parse_seq(tkr):
    cs = []
    while 1:
        lex, val = next(tkr)
        if lex is None or lex in RSEP:
            break
        elif lex in LSEP:
            cs.append(parse_seq(tkr))
        else:
            cs.append(val)
    return tuple(cs)

def parse_one(tkr):
    lex, val = next(tkr)
    if lex in LSEP:
        return parse_seq(tkr)
    elif lex == "LAM":
        next(tkr)               # discard left paren
        return parse_lam(tkr)
    else:
        return val

parse_one(tokenize(inp)) 

def parse(inp):
    return parse_one(tokenize(inp))

parse(inp)
