import preamble
from metaparse import *
from pprint import pprint

class S(metaclass=LALR.meta):
# class S(metaclass=GLR.meta):
    a, b, c = 'abc'
    def S(A, B, C): return (A, *B, C)
    def A(a): return a
    def A(): return ()
    def B(): return ()
    def B(B, b): return B + (b,)
    def C(c): return c


# pprint([*p.lexer.tokenize('abbbc', True)])

from unittest import main, TestCase

class Test(TestCase):
    def test(self):
        r = S.interpret('abbbbc')
        self.assertEqual(r, ('a', 'b', 'b', 'b', 'b', 'c'))


# pprint(p.__dict__)

if __name__ == '__main__':
    main()
