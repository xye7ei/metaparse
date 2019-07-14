import unittest

from metaparse import Grammar, LALR

class GArith(metaclass=LALR.meta):

    'Textbook Grammar.meta for simple arithmetics.'

    # E -> E + T
    # E -> T
    # T -> T * F
    # T -> F
    # F -> NUMBER
    # F -> ( E )

    IGNORED = r' '

    plus   = r'\+'
    times  = r'\*'

    def number(lex: r'\d+'):
        return int(lex)

    left   = r'\('
    right  = r'\)'


    def Expr(Expr, plus, Term):
        return Expr + Term
    def Expr(Term):
        return Term

    def Term(Term, times, Factor):
        return Term * Factor
    def Term(Factor):
        return Factor

    def Factor(number):
        return number
    def Factor(left, Expr, right):
        return Expr

    # def Atom(number):
    #     return int(number)

g = Grammar(GArith.rules)
p = GArith

# l = p.lexer
# print(list(l.tokenize('1 2')))
# assert 0

class TestArithParser(unittest.TestCase):

    def test_FIRST(self):
        self.assertEqual(g.FIRST['Expr'], {'left', 'number'})
        self.assertEqual(g.FIRST['Term'], {'left', 'number'})
        self.assertEqual(g.FIRST['Factor'], {'left', 'number'})
        self.assertEqual(g.FIRST['number'], {'number'})

    def test_single(self):
        inp = '0'
        self.assertEqual(eval(inp), p.interpret(inp))

    def test_normal(self):
        inp = '3 + 2 * (5 + 11) * 2 + 3'
        self.assertEqual(eval(inp), p.interpret(inp))

    def test_tough(self):
        inp = '3 + 2 * (5 + 11)'
        tough_inp = ' + '.join(inp for _ in range(100))
        self.assertEqual(eval(inp), p.interpret(inp))


if __name__ == '__main__':

    unittest.main()

    # For debugging
    # t = TestArithParser()
    # t.test_normal()

    # tough = ' + '.join(['(2 * (1 + 1) + 2 * 2)'] * 1000)
    # %timeit ari_LALR.meta.interpret(tough)
    # 1 loops, best of 3: 347 ms per loop

    # with open('C:/Users/Shellay/Desktop/ari.psr', 'wb') as o:
    #     o.write(ari_LALR.meta.dumps())
