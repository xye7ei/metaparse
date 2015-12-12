from grammar import cfg
from earley import Earley
from lalr import LALR

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

    # def Factor(Atom):
    #     return Atom 
    def Factor(Atom):
        return Atom
    def Factor(left, Expr, right):
        return Expr

    def Atom(number):
        return int(number)

p = Earley(GArith)
q = LALR(GArith)

inp = '3 + 2 * (5 + 11)'

tough_inp = ' + '.join(inp for _ in range(10000))

if __name__ == '__main__':

    import pprint as pp

    s = p.parse(inp)
    t = q.parse(inp)
    r = q.interprete(inp)

    # # print(inp)
    # pp.pprint(s)
    pp.pprint(t)
    pp.pprint(r)

    # %timeit q.parse(tough_inp) 
    # 1 loops, best of 3: 3.07 s per loop
    # This is 133% of the speed compared to `ply` package.

    # %timeit q.interprete(tough_inp)
    # 1 loops, best of 3: 3.37 s per loop
    # Interpretation is slightly longer than parsing.

    # pp.pprint(q.Ks)
    # pp.pprint(q.ACTION)

    pass
