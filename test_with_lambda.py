"""
A simple haskell-like grammar for basic lambda-calculus
"""

from metaparse import *

class Lam(metaclass=cfg):
# class Lam(metaclass=LALR.meta):

    # IGNORED = r'(^ *\n$)|[ ]' 
    # IGNORED = r'^\s+'
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
        return [expr_1, expr_2]
    def appl(appl, expr):
        return [appl, expr]

    # Lambda-Abstraction
    def abst(LAMBDA, varlist, ARROW, expr):
        return (varlist, expr)
    def varlist(VAR):
        return {VAR}
    def varlist(varlist, COMMA, VAR):
        return {*varlist, VAR}

    # Let-expression with environmental bindings
    def let(LET, binds, IN, exprx):
        return ['LET', binds, exprx]
    def bind(pat, EQ, exprx):
        return {pat: exprx}
    def binds(bind):
        return bind
    def binds(binds, SEMI, bind):
        return {**binds, **bind}


psr = Earley(Lam)
psr = GLR(Lam)

# Test whether the grammar is LALR to exclude potential ambiguity
# and prepare for better performance
psr = LALR(Lam)


inp = """
let a = 3 ;
    P q = z x y
 in 
   add a b y
"""

inp = """
let a = 1
in (let b = 2 in f a b)
"""

# print(Lam)

pp.pprint(list(psr.grammar.tokenize(inp, False)))

# pp.pprint(psr.recognize(inp))
# pp.pprint(res)
# pp.pprint(psr.parse(inp))
# x1, x2 = psr.parse(inp)
# assert x1.to_tuple() == x2.to_tuple()
pp.pprint(psr.interpret(inp))
