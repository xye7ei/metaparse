from unittest import TestCase, main
from metaparse import *


class LangExpr(metaclass=GLR.meta):

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

        p = LangExpr.prepare_generalized()

        inp = '1 + 2 * 3 + 4'
        x = list(LangExpr.lexer.tokenize(inp))

        next(p)
        for tk in x:
            r = p.send(tk)
        else:
            r = p.send(END_TOKEN)
            # 5 combinations for association!
            # 
            # How to calc the number of combinations?
            # 
            # <==>
            # 
            # Given i operators, how many binary trees can they form
            # with the same infix-order?
            #
            # - choose each one as a subtree root
            # - divide by the root, calc recursively
            # 
            # B(0) == 1
            # B(1) == 1
            # B(2) == B(1) + B(1) == 2
            # B(3) == B(2) + B(1)*B(1) + B(2) == 2 + 1 + 2 == 5
            # B(4) == B(3) + B(1)B(2) + B(2)B(1) + B(3) == 5 + 2 + 2 + 5 == 14
            # ...
            # B(n) == sum(B(i)B(n-1-i) for i in [1..n-1])
            self.assertEqual(len(r), 2 + 1 + 2)

    def test_send_more(self):
        inp = '1 + 2 * 3 + 4 * 5'
        y = LangExpr.interpret_generalized(inp)
        self.assertEqual(len(y), 14)

    def test_parse(self):
        y = LangExpr.interpret_generalized('1 + 2 * 3')
        self.assertEqual(y, [9, 7])


if __name__ == '__main__':
    main()

