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

p = Earley(GArith)
q = LALR(GArith)
g = GLR(GArith)

inp = '3 + 2 * (5 + 11)'

tough_inp = ' + '.join(inp for _ in range(10000))

if __name__ == '__main__':

    import pprint as pp

    p.parse(inp)

    q.parse(inp)
    q.interprete(inp)

    g.parse(inp)

    # print(inp)
    # pp.pprint(s)
    # pp.pprint(t)
    # pp.pprint(r)

    # %timeit p.parse(tough_inp) 
    # 1 loops, best of 3: 12.7 s per loop

    # %timeit q.parse(tough_inp) 
    # 1 loops, best of 3: 3.07 s per loop
    # This is 133% of the speed compared to LALR parser in
    # `ply` package.

    y = g.parse(tough_inp) 
    # %timeit g.parse(tough_inp) 
    # 1 loops, best of 3: 3.46 s per loop
    # This is comparable to LALR parser.

    # %timeit q.interprete(tough_inp)
    # 1 loops, best of 3: 3.37 s per loop
    # Interpretation is slightly longer than parsing.

    # pp.pprint(q.Ks)
    # pp.pprint(q.ACTION)

    pass
