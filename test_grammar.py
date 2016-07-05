import pprint as pp
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


class Glrval(metaclass=cfg): 
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
        return 'id' 
    def R(L):
        return L 


# pp.pprint(Glrval.closure_with_lookahead(G.make_item(0, 0), '%'))
Plrval = Earley(Glrval)
Rlrval = GLR(Glrval)
Qlrval = LALR(Glrval)

pp.pprint(Plrval.interpret('abc = * * ops'))
pp.pprint(Plrval.interpret('* abc = * * * ops'))

pp.pprint(Rlrval.interpret('abc = * * ops'))
pp.pprint(Rlrval.interpret('* abc = * * * ops'))

# pp.pprint(Qlrval.interpret('abc = * * ops'))
# pp.pprint(Qlrval.interpret('* abc = * * * ops'))

class GArith(metaclass=cfg): 
    IGNORED = r'\s+' 
    plus   = r'\+'
    times  = r'\*'
    number = r'\d+'
    left   = r'\('
    right  = r'\)' 
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

