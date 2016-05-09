import preamble

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

if __name__ == '__main__':

    s = Gchurch.parse('succ succ succ zero')
    v = Gchurch.interprete('succ succ succ zero')

    import pprint as pp
    pp.pprint(s)

    assert Gchurch.interprete('zero') == 0
    assert Gchurch.interprete('succ zero') == 1
    assert Gchurch.interprete('succ succ zero') == 2
    assert Gchurch.interprete('succ succ succ zero') == 3
