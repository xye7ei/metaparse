import preamble
import unittest

from metaparse import *

from pprint import pprint


class TestLRGrammar(unittest.TestCase):

    def test_LALR_report(self):
        """LALR parser should report conflicts for ambiguous Grammar.meta! """
        with self.assertRaises(LanguageError):

            class Gif(metaclass=LALR.meta):

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

    # def test_many(self):

    #         class Gif(metaclass=LALR.meta):

    #             IF     = r'if'
    #             THEN   = r'then'
    #             ELSE   = r'else'
    #             EXPR   = r'e'
    #             SINGLE = r's'

    #             def stmt(ifstmt):
    #                 return ifstmt 

    #             def stmt(SINGLE):
    #                 return SINGLE 

    #             def ifstmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
    #                 return ('ite', EXPR, stmt_1, stmt_2) 

    #             def ifstmt(IF, EXPR, THEN, stmt):
    #                 return ('it', EXPR, stmt)


if __name__ == '__main__':

    unittest.main()
