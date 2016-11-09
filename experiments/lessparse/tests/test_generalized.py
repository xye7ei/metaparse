import preamble
from unittest import TestCase, main
from metaparse import *


class pExpr(metaclass=GLR.meta):

    'An ambigious grammar for arithmetic expressions.'

    def plus(lex: r'\+'):
        return lex

    def times(lex: r'\*'):
        return lex

    def number(lex: r'\d+'):
        return int(lex)


    def expr(expr, plus, expr_1):
        return expr + expr_1

    def expr(expr, times, expr_1):
        return expr * expr_1

    def expr(number):
        return number


class Test(TestCase):

    def test_send(self):

        p = pExpr.prepare_generalized()

        inp = '1 + 2 * 3 + 4'
        x = list(pExpr.lexer.tokenize(inp))
        # print(len(x), x)

        next(p)
        for tk in x:
            r = p.send(tk)
            # print(r)
        else:
            r = p.send(None)
            # print(r)
            self.assertEqual(len(r), 1)
            r_last = r[len(x)]
            self.assertEqual(len(r_last), (2 + 1 + 2))


    def test_parse(self):
        y = pExpr.interpret_generalized('1 + 2 * 3')
        self.assertEqual(y[5], [9, 7])

if __name__ == '__main__':
    # print(pExpr.__doc__)
    main()

