import preamble

from metaparse import cfg, Earley, GLR, LALR
from pprint import pprint

class S(metaclass=cfg):
# class S(metaclass=GLR.meta):
    a, b, c = 'abc'
    def S(A, B, C): return (A, *B, C)
    def A(a): return a
    def A(): return ()
    def B(): return ()
    def B(B, b): return B + (b,)
    def C(c): return c

# pprint(S.first_of_seq(('B', 'b')))
pprint(S.FIRST['A'])
pprint(S.FIRST['B'])
pprint(S.FIRST['C'])
# assert 0

# p = LALR(S)
p = Earley(S)

# pprint([*p.lexer.tokenize('abbbc', True)])

from unittest import main, TestCase

class Test(TestCase):
    def test(self):
        # r = S.parse('abbbc')
        r = p.interpret_many('abbbbc')

        assert r == [('a', 'b', 'b', 'b', 'b', 'c')]
        pprint(r)

# pprint(p.__dict__)

if __name__ == '__main__':
    main()
