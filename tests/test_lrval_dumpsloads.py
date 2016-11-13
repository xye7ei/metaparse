import preamble

from metaparse import *

table = []
refs = 0

class G(metaclass=LALR.meta):

    EQ   = r'='

    def STAR(lex: r'\*'):
        global refs
        refs += 1
        return lex

    def ID(lex: r'[_a-zA-Z]\w*'):
        table.append(lex)
        return lex

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


import unittest

class TestDumpLoad(unittest.TestCase):

    def test_dumpload(self):

        inp = '*a = **b'

        import pprint as pp

        p1 = G

        s1 = p1.dumps()
        p1 = LALR.loads(s1, globals()) 
        s1 = p1.dumps()
        p1 = LALR.loads(s1, globals()) 

        r = p1.interpret(inp)
        r = p1.interpret(inp)

        self.assertEqual(r, ('assign',
                             ('deref', 'a'),
                             ('deref', ('deref', 'b'))))

        self.assertEqual(table, ['a', 'b', 'a', 'b'])
        self.assertEqual(refs, 6)

        # s = p1.lexer.dumps()
        # lexer = Lexer.loads(s, globals())
        # xs = list(lexer.tokenize(inp, True))
        # # pp.pprint(xs)

        # self.assertEqual(refs, 9)
        # # pp.pprint(p1)
        # # pp.pprint(p1.__dict__)


if __name__ == '__main__':

    unittest.main()
