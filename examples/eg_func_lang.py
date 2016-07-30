"""A simple haskell-like grammar for basic lambda-calculus.

A GLR(0) parser can be used to find LR(0)-conflicts as well as
contruct partial parse trees.

"""

import preamble
import pprint as pp

from metaparse import *

from collections import namedtuple as data

Let   = data('Let', 'binds body')
Abst  = data('Abst', 'par body')
Appl  = data('Appl', 'func arg')
Var   = data('Var', 'symbol')
Value = data('Value', 'value')

Expr  = [Let, Abst, Appl, Var, Value]
for E in Expr:
    E.__repr__ = tuple.__repr__

TW = r'[ \t]*'
TWN = r'[ \t\n]*'

class Lam(metaclass=cfg):

    # IGNORED = r'(^[ \t]*\n)| '
    # NEWLINE = r'\n'         # + TWN
    IGNORED = r'\s+'

    EQ      = r'='          # + TW
    IN      = r'in'         # + TWN
    LET     = r'let'        # + TWN
    LAMBDA  = r'\\'         # + TWN
    ARROW   = r'->'         # + TWN
    COMMA   = r','          # + TW
    SEMI    = r';'          # + TW
    L1      = r'\('         # + TW
    R1      = r'\)'         # + TW

    VALUE   = r'\d+'        # + TW
    VAR     = r'[_a-z]\w*'  # + TW
    CONS    = r'[A-Z]\w*'   # + TW

    INFIX   = r'[\+\-\*\/]' # + TW

    def prog(binds):
        return binds

    # Stand-alone expression
    def exprx(expr):
        return expr
    def exprx(let):
        return let
    def exprx(abst):
        return abst
    def exprx(appl):
        return appl
    # def exprx(L1, exprx, R1):
    #     return exprx

    # Singleton expression
    def expr(VALUE):
        return float(VALUE)
    def expr(VAR):
        return VAR
    def expr(L1, exprx, R1):
        return exprx

    # Pattern
    def pat(VAR):
        return VAR
    def pat(CONS, arglist):
        return (CONS, arglist)
    def arglist(arglist, expr):
        return arglist + (expr,)
    def arglist():
        return ()

    # Application (Curried)
    def appl(expr_1, expr_2):
        return Appl(expr_1, expr_2)
    def appl(appl, expr):
        return Appl(appl, expr)
    def appl(expr_1, INFIX, expr_2):
        return Appl(INFIX, expr_1, expr_2)

    # Lambda-Abstraction
    def abst(LAMBDA, parlist, ARROW, exprx):
        tar = exprx
        for par in reversed(parlist):
            tar = Abst(par, tar)
        return tar
    def parlist(VAR):
        return [VAR]
    def parlist(parlist, COMMA, VAR):
        return [*parlist, VAR]

    # Let-expression with environmental bindings
    def let(LET, binds, IN, exprx):
        return Let(binds, exprx)
    def bind(pat, EQ, exprx):
        return {pat: exprx}
    def binds(bind):
        return bind
    def binds(binds, SEMI, bind):
        return {**binds, **bind}

    # def _env():
    #     print('Env!')

    # def _unify():
    #     print('Unify!')


# Test whether the grammar is LALR to exclude potential ambiguity
# and prepare for better performance
psr_ear = Earley(Lam)
psr_gll = GLL(Lam)
psr_glr = GLR(Lam)
psr_lalr = LALR(Lam)


inp = """
let a = 3 ;
    P q = u v #
 in  #
   map (\c, d -> f c d) xs ys
"""

inp = """

k = let
     a = 3 ;
     P p q = u v
 in 
   map (\c, d -> f c d) xs ys ;

l = 3 ;
m = 4
"""


# print(Lam)
# psr_gll.interpret(inp) # LEFT-RECURSION!!!!
# psr_glr.interpret(inp)
# psr_lalr.interpret(inp)

psr = psr_lalr
# psr = psr_glr
# psr = psr_ear

tough_inp = '   ;\n'.join([inp for _ in range(100)])

# pp.pprint(list(psr.grammar.tokenize(inp, False)))
# pp.pprint(psr.interpret_many(inp))
# print(len(psr.ACTION))
# pp.pprint(psr.ACTION)

pp.pprint(psr_lalr.interpret(tough_inp))
