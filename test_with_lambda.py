"""A simple haskell-like grammar for basic lambda-calculus.

A GLR(0) parser can be used to find LR(0)-conflicts as well as
contruct partial parse trees.

"""

from metaparse import *

from collections import namedtuple as data


Let   = data('Let', 'binds body')
Abst  = data('Abst', 'par body')
Appl  = data('Appl', 'func arg')
Var   = data('Var', 'symbol')
Value = data('Value', 'value')


class Lam(metaclass=cfg):

    IGNORED = r'\s+'

    EQ     = r'='
    IN     = r'in'
    LET    = r'let'
    LAMBDA = r'\\'
    ARROW  = r'->'
    COMMA  = r','
    SEMI   = r';'
    L1     = r'\('
    R1     = r'\)'

    VALUE   = r'\d+'
    VAR     = r'[_a-z]\w*'
    CONS    = r'[A-Z]\w*'

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
        # return ['LET', binds, exprx]
        return Let(binds, exprx)
    def bind(pat, EQ, exprx):
        return {pat: exprx}
    def binds(bind):
        return bind
    def binds(binds, SEMI, bind):
        return {**binds, **bind}

    def _env():
        print('Env!')

    def _unify():
        print('Unify!')


psr = Earley(Lam)
psr = GLR(Lam)

# Test whether the grammar is LALR to exclude potential ambiguity
# and prepare for better performance
psr = LALR(Lam)


inp = """
let a = 3 ;
    P q = u v #
 in  #
   map (\c, d -> f c d) xs ys
"""

# inp = """
# let a = 1
# in (let b = 2 in f a b)
# """

# print(Lam)

pp.pprint(list(psr.grammar.tokenize(inp, False)))
pp.pprint(psr.interpret(inp))
