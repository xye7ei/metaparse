import pprint as pp
import metaparse

from metaparse import *


class G(metaclass=cfg):
    a = r'a'
    b = r'b'
    def S(S, A, B): return
    def S()       : return
    def A(a)      : return
    def A()       : return
    def B(b)      : return

# pp.pprint(G.first_seq(G.rules[0].rhs, '#'))


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
e_LRVal = Earley(LRVal)
g_LRVal = GLR(LRVal)
l_LRVal = LALR(LRVal)

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

print('Finished.')

metaparse
