import preamble

from metaparse import *

table = []
refs = 0

class Glrval(metaclass=LALR.meta):

    EQ   = r'='

    STAR = r'\*'
    def STAR(lex: r'\*') -> 3:
        global refs
        refs += 1
        # return lex

    # ID   = r'[_a-zA-Z]\w*'
    def ID(lex: r'[_a-zA-Z]\w*') -> 3:
        table.append(lex)
        return lex

    # def ID(ID = r'[_a-zA-Z]\w*') -> 3:
        # table.add(ID)
        # return ID

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


if __name__ == '__main__':

    inp = '*a = **b'

    import pprint as pp

    result = Glrval.interpret(inp)
    assert result == \
        ('assign',
         ('deref', 'a'),
         ('deref', ('deref', 'b')))

    assert table == ['a', 'b'], table
    assert refs == 3

