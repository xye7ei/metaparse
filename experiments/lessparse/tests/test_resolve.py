import preamble
from metaparse import LALR
from pprint import pprint
from unittest import TestCase, main 


class p_ite(metaclass=LALR.meta):

    IGNORED = r'[ \(\)]'
    IF     = r'if'
    THEN   = r'then', 1
    ELSE   = r'else', 2
    EXPR   = r'e'
    SINGLE = r's'

    def stmt(ifstmt):
        return ifstmt 

    def stmt(SINGLE):
        return SINGLE 

    def ifstmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return ('ite', stmt_1, stmt_2) 

    def ifstmt(IF, EXPR, THEN, stmt):
        return ('it', stmt)



class Test(TestCase):

    def test_parse(self):

        inp = 'if e then (if e then (if e then s else s) else s)'
        res = p_ite.interpret(inp)
        
        self.assertEqual(res, ('it', ('ite', ('ite', 's', 's'), 's')))


if __name__ == '__main__':
    main()
