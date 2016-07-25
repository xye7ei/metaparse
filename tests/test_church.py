import preamble

from metaparse import LALR

class Gchurch(metaclass=LALR.meta):

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

    assert Gchurch.interpret('zero')                == 0
    assert Gchurch.interpret('succ zero')           == 1
    assert Gchurch.interpret('succ succ zero')      == 2
    assert Gchurch.interpret('succ succ succ zero') == 3
