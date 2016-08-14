import preamble

from metaparse import LALR
from collections import namedtuple

Succ = namedtuple('Succ', 'value')
Add = namedtuple('Add', 'left right')


class Peano(metaclass=LALR.meta):

    L = r'\('
    R = r'\)'
    COMMA = r','

    def ZERO(lex: r'0'):
        return 0
    def S(lex: r'S'):
        return lex
    def ADD(lex: r'\+'):
        return lex

    def expr(nat):
        return nat
    def expr(add):
        return add

    def nat(ZERO):
        return ZERO
    def nat(S, L, nat, R):
        return Succ(nat)

    def add(ADD, L, nat_1, COMMA, nat_2, R):
        assert isinstance(nat_1, (int, Succ))
        # if nat_1 == 0:
        #     return nat_2
        # else:
        #     m = nat_1.value
        #     return Succ(Add(m, nat_2))
        return Add(nat_1, nat_2)


inp = '+(S(S(0)), S(0))'

r = Peano.interpret(inp)
print(r)
