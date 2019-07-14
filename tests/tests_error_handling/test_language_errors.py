from metaparse import LanguageError, LALR
import unittest


class TestLangError(unittest.TestCase):
    
    def test_missing_symbol(self):
        with self.assertRaises(LanguageError) as excCtx:

            class ExprLang(metaclass=LALR.meta):

                NUM = '\d+'
                PLUS = '\+'
                # TIMES = '\*'

                def expr(expr, PLUS, term):
                    return expr + term

                def expr(expr, TIMES, term):
                    return expr * term

                def expr(term):
                    return term

                def term(NUM):
                    return int(NUM)

                def factor(NUM):
                    return int(NUM)

        self.assertIn(
            'No lexical pattern provided for terminal symbol: TIMES',
            excCtx.exception.message)

    def test_unreachable_rule(self):
        with self.assertRaises(LanguageError) as excCtx:

            class ExprLang(metaclass=LALR.meta):

                NUM = '\d+'
                PLUS = '\+'
                TIMES = '\*'

                def expr(expr, PLUS, term):
                    return expr + term

                def expr(expr, TIMES, term):
                    return expr * term

                def expr(term):
                    return term

                def term(NUM):
                    return int(NUM)

                def factor(NUM):
                    return int(NUM)

        self.assertIn(
            "There are unreachable nonterminals at 5th rule: {'factor'}.",
            excCtx.exception.message)

    
class TestLangErrorApi2(unittest.TestCase):

    def test_missing_symbol(self):
        with self.assertRaises(LanguageError) as excCtx:
            p = LALR()
            with p as (lex, rule):
                lex(a = 'a')
                lex(b = 'b')
                @rule
                def S(a, S, b): pass
                @rule
                def S(): pass
                @rule
                def S(c): pass
        self.assertIn(
            'No lexical pattern provided for terminal symbol: c',
            excCtx.exception.message)

    def test_unreachable_rule(self):
        with self.assertRaises(LanguageError) as excCtx:
            p = LALR()
            with p as (l, r):
                l(a = 'a')
                l(b = 'b')
                @r
                def S(a, S, b): pass
                @r
                def S(): pass
                @r
                def B(a): pass
                @r
                def B(b): pass

        self.assertIn(
            "There are unreachable nonterminals at 3th rule: {'B'}.",
            excCtx.exception.message)


if __name__ == '__main__':
    unittest.main()
