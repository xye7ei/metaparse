from sys import path; path.append('..')

from grammar import cfg
from earley import Earley
from lalr import LALR
from glr import GLR

class GArith(metaclass=cfg):

    # E -> E + T
    # E -> T
    # T -> T * F
    # T -> F
    # F -> NUMBER
    # F -> ( E )

    IGNORED = r' '

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

    
ari_earl = Earley(GArith)
ari_lalr = LALR(GArith)
ari_glr = GLR(GArith)

inp = '3 + 2 * (5 + 11)'

tough_inp = ' + '.join(inp for _ in range(10000))


if __name__ == '__main__':

    from pprint import pprint as pp

    ps1 = ari_earl.parse(inp)
    p2 = ari_lalr.parse(inp)
    ps3 = ari_glr.parse(inp)

    assert p2 == ps3[0], \
        'Result from LALR should be equal to the first result from GLR'

    assert 1 == ari_lalr.interprete('1')
    assert 3 == ari_lalr.interprete('1 + 2')
    assert 35 == ari_lalr.interprete('3 + 2 * (5 + 11)')
    assert 135 == ari_lalr.interprete('3 + 2 * (5 + 11) + ((100))')

    # print(inp)
    # pp.pprint(s)
    # pp.pprint(t)
    # pp.pprint(r)

    # %timeit ari_earl.parse(tough_inp)
    # 1 loops, best of 3: 12.7 s per loop

    # %timeit ari_lalr.parse(tough_inp)
    # 1 loops, best of 3: 3.07 s per loop
    # This is 133% of the speed compared to LALR parser in `ply` package.

    y = ari_glr.parse(tough_inp)
    # %timeit ari_glr.parse(tough_inp)
    # 1 loops, best of 3: 3.46 s per loop
    # This GLR parser is comparable to LALR parser in this
    # case.

    # %timeit ari_glr.interprete(tough_inp)
    # 1 loops, best of 3: 3.37 s per loop
    # Interpretation is slightly more time-consuming than
    # parsing.

    # pp.pprint(ari_lalr.Ks)
    # pp.pprint(ari_lalr.ACTION)

    pass
