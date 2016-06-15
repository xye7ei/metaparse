import preamble

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


if __name__ == '__main__':

    inp = '*id = **id'
    Glrval.parse(inp)

    import pprint as pp

    pp.pprint(Glrval.parse(inp))

    result = Glrval.interpret(inp)
    assert result == \
        ('assign',
         ('deref', 'id'),
         ('deref', ('deref', 'id')))
