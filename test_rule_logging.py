from metaparse import *

@cfg2
class S:

    L1 = r'\('
    R1 = r'\)'
    SYMBOL = r'[^\(\)\s]+'

    @rule
    def sexp(SYMBOL):
        return SYMBOL

    @rule
    def sexp(L1, slist, R1):
        return slist

    @rule
    def slist():
        return ()

    @rule
    def slist(sexp, slist):
        return (sexp,) + slist



psr = GLR(S)
psr = LALR(S)

inp = """
(  (a b) c (d )  e )
"""

res = psr.interpret(inp)

pp.pprint(res)

