from earley import earley

class Glrval(metaclass=earley):

    EQ   = r'='
    STAR = r'\*'
    ID   = r'id'


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

inp = '*id = **id'
Glrval.parse(inp)

import pprint as pp

pp.pprint(Glrval.parse(inp))
pp.pprint(Glrval.eval(inp))
