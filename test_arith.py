from earley import earley

class GArith(metaclass=earley):

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

# E -> E + T
# E -> T
# T -> T * F
# T -> F
# F -> NUMB
# F -> ( E )

inp = '3 + 2 * 5 + 11'
s = GArith.parse(inp)
fin = s[-1][-1][0]
# print(inp)
import pprint as pp
pp.pprint(s)
