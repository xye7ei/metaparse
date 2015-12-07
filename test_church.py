from earley import earley

class Gchurch(metaclass=earley):

    """
    Grammar for interpreting Church-Numerals.
    """

    ZERO = r'zero'
    SUCC = r'succ'

    def num(ZERO):
        return 0


    def num(SUCC, num):
        return num + 1

s = Gchurch.parse('succ succ succ zero')
v = Gchurch.eval('succ succ succ zero')
import pprint as pp
pp.pprint(s)
pp.pprint(v)
