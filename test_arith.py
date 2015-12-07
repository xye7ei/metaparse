from grammar import cfg
from earley import Earley

class GArith(metaclass=cfg):

    # E -> E + T
    # E -> T
    # T -> T * F
    # T -> F
    # F -> NUMB
    # F -> ( E )

    PLUS  = r'\+'
    TIMES = r'\*'
    NUMB  = r'\d+'
    LEFT  = r'\('
    RIGHT = r'\)'


    def expr(expr, PLUS, term):
        return expr + term

    def expr(term):
        return term

    def term(term, TIMES, factor):
        return term * factor

    def term(factor):
        return factor

    def factor(atom):
        return atom

    def factor(LEFT, expr, RIGHT):
        return expr

    def atom(NUMB):
        return int(NUMB)

p = Earley(GArith)

inp = '3 + 2 * 5 + 11'

s = p.parse(inp)

# print(inp)
import pprint as pp
pp.pprint(s)
