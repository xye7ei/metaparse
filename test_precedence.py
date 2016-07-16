from metaparse import cfg, GLR, LALR

class E(metaclass=cfg):

    num = r'\d+'

    sup = r'\*\*', 3            # r'\*\*' is matched before r'\*'

    mul = r'\*', 2
    div = r'\/', 2

    add = r'\+', 1
    mns = r'-', 1

    l   = r'\('
    r   = r'\)'

    def E(E, add, E_1):
        return '({} + {})'.format(E, E_1)
    def E(E, mns, E_1):
        return '({} - {})'.format(E, E_1)
    def E(E, mul, E_1):
        return '({} * {})'.format(E, E_1)
    def E(E, div, E_1):
        return '({} / {})'.format(E, E_1)
    def E(E, sup, E_1):
        return '({} ** {})'.format(E, E_1)
    def E(num):
        return num
    def E(l, E, r):
        return E


import pprint as pp

# pp.pprint(E.parse_many('3 + 2 * 7'))
# pp.pprint(E.parse_many('3 + 2 * 7 + 1'))
# pp.pprint(E.interpret_many('3 + 2 * 7 + 1'))

print(E)
pp.pprint(E.OP_PRE)
psr = LALR(E)
# print(psr.table.__len__())
# pp.pprint([*zip(psr.Ks, psr.ACTION)])

# print(psr.interpret('3 + 2 * 7'))
# print(psr.interpret('3 * 2 + 7'))
print(psr.interpret('3 + 2 * 7 / 5 - 1'))
print(psr.interpret('3 + 2 * 7 ** 2 * 5'))
