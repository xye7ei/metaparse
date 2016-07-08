import pprint as pp

from metaparse import *


class LRVal(metaclass=cfg): 
    IGNORED = r'\s+'
    EQ   = r'='
    STAR = r'\*'
    ID   = r'\w+'

    def S(L, EQ, R):
        return ('assign', L, R) 
    def S(R):
        return ('expr', R) 
    def L(STAR, R):
        return ('deref', R) 
    def L(ID):
        return ID
    def R(L):
        return L 


# pp.pprint(Glrval.closure_with_lookahead(G.make_item(0, 0), '%'))
# e_LRVal = Earley(LRVal)
# g_LRVal = GLR(LRVal)
# l_LRVal = LALR(LRVal)

# pp.pprint(e_LRVal.interpret('abc = * * ops'))
# pp.pprint(e_LRVal.interpret('* abc = * * * ops'))

# pp.pprint(g_LRVal.interpret('abc = * * ops'))
# pp.pprint(g_LRVal.interpret('* abc = * * * ops'))

# pp.pprint(l_LRVal.interpret('abc'))
# pp.pprint(l_LRVal.interpret('abc = * * ops'))
# pp.pprint(l_LRVal.interpret('* abc = * * * ops'))


class GArith(metaclass=cfg): 
    IGNORED = r'\s+' 
    plus   = r'\+'
    times  = r'\*'
    number = r'\d+'
    left   = r'\('
    right  = r'\)' 
    # bla    = r'bla'
    def Expr(Expr, plus, Term):
        return Expr + Term
    def Expr(Term):
        return Term 
    def Term(Term, times, Factor):
        return Term * Factor
    def Term(Factor):
        return Factor 
    def Factor(Atom):
        return Atom
    def Factor(left, Expr, right):
        return Expr 
    def Atom(number):
        return int(number)


class SExp(metaclass=cfg):
    SINGLE = r'[^ \(\)\[\]\{\}]+'
    L1 = r'\('
    R1 = r'\)'
    def sexp(SINGLE):
        return SINGLE
    def sexp(L1, slist, R1):
        return slist
    def slist(sexp, slist):
        return [sexp, *slist]
    def slist():
        return []


# pp.pprint(SExp)
