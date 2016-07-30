import preamble
import unittest

from metaparse import *

class GArith(metaclass=cfg):

    'Textbook grammar for simple arithmetics.'

    # E -> E + T
    # E -> T
    # T -> T * F
    # T -> F
    # F -> NUMBER
    # F -> ( E )

    IGNORED = r' '

    plus   = r'\+'
    times  = r'\*'
    # number = r'\d+'
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


ari_ear = Earley(GArith)
ari_lalr = LALR(GArith)
# Naive GLL still cannot handle left-recursion
# ari_gll = GLR(GArith)
ari_glr = GLR(GArith)
ari_gll = GLL(GArith)


class TestArithParser(unittest.TestCase):

    def test_FIRST(self):
        self.assertEqual(GArith.FIRST['Expr'], {'left', 'number'})

    def test_single(self):
        inp = '0'
        ps1 = ari_ear.interpret_many(inp)
        ps2 = ari_lalr.interpret_many(inp)
        ps3 = ari_glr.interpret_many(inp)
        self.assertEqual(ps1, ps2)
        self.assertEqual(ps2, ps3)

    def test_normal(self):
        inp = '3 + 2 * (5 + 11) * 2 + 3'
        ps0 = [eval(inp)]
        ps1 = ari_ear.interpret_many(inp)
        ps2 = ari_lalr.interpret_many(inp)
        ps3 = ari_glr.interpret_many(inp)
        self.assertEqual(ps0, ps1)
        self.assertEqual(ps1, ps2)
        self.assertEqual(ps2, ps3)

    def test_tough(self):
        inp = '3 + 2 * (5 + 11)'
        tough_inp = ' + '.join(inp for _ in range(100))
        ps0 = [eval(tough_inp)]
        ps1 = ari_ear.interpret_many(tough_inp)
        ps2 = ari_lalr.interpret_many(tough_inp)
        ps3 = ari_glr.interpret_many(tough_inp)
        self.assertEqual(ps0, ps2)
        self.assertEqual(ps1, ps2)
        self.assertEqual(ps2, ps3)


if __name__ == '__main__':

    unittest.main()

    # For debugging
    # t = TestArithParser()
    # t.test_normal()

