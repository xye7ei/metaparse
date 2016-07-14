from metaparse import GLR, LALR

class E(metaclass=LALR.meta):
# class E(metaclass=GLR.meta):
    num = r'\d+'
    plus = r'\+'
    times = r'\*'
    def E(E, plus, E_1):
        # return E + E
        return '({} + {})'.format(E, E_1)
    def E(E, times, E_1):
        # return E * E
        return '({} * {})'.format(E, E_1)
    def E(num):
        # return int(num)
        return num

import pprint as pp

# pp.pprint(E.parse_many('3 + 2 * 7'))
# pp.pprint(E.parse_many('3 + 2 * 7 + 1'))
pp.pprint(E.interpret_many('3 + 2 * 7 + 1'))
