import unittest

from metaparse import LanguageError, LALR, GLR

from pprint import pprint


class TestLRGrammar(unittest.TestCase):

    def test_LALR_report(self):
        """LALR parser should report conflicts for ambiguous Grammar.meta! """
        with self.assertRaises(LanguageError) as caught:

            class LangIfThenElse(metaclass=LALR.meta):

                IF     = r'if'
                THEN   = r'then'
                ELSE   = r'else'
                EXPR   = r'e'
                SINGLE = r's'

                def stmt(ifstmt):
                    return ifstmt 

                def stmt(SINGLE):
                    return SINGLE 

                def ifstmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
                    return ('ite', EXPR, stmt_1, stmt_2) 

                def ifstmt(IF, EXPR, THEN, stmt):
                    return ('it', EXPR, stmt)

        self.assertIn(
            'Conflict on lookahead: ELSE',
            caught.exception.message)

    def test_many(self):

        class LangIfThenElse(metaclass=GLR.meta):

            IF     = r'if'
            THEN   = r'then'
            ELSE   = r'else'
            EXPR   = r'\d'
            SINGLE = r'[xyz]'

            def stmt(ifstmt):
                return ifstmt 

            def stmt(SINGLE):
                return SINGLE 

            def ifstmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
                return ('ite', EXPR, stmt_1, stmt_2) 

            def ifstmt(IF, EXPR, THEN, stmt):
                return ('it', EXPR, stmt)

        results = LangIfThenElse.interpret_generalized('if 1 then if 2 then x else y')
        self.assertEqual(len(results), 2)
        self.assertIn(
            ('it', '1', ('ite', '2', 'x', 'y')),
            results)
        self.assertIn(
            ('ite', '1', ('it', '2', 'x'), 'y'),
            results)


if __name__ == '__main__':
    unittest.main()
