import preamble
from metaparse import LALR
from pprint import pprint
from unittest import TestCase, main 


class p_ite(metaclass=LALR.meta):

    'Dangling else grammar with ambiguity resolved by precedence.'

    IGNORED = r'[ \(\)]'
    IF     = r'if'
    THEN   = r'then', 1
    def ELSE(lex: r'else') -> 2:
        return lex

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

    # inp = 'if e then else (if e then (if e then s else s) else s)'
    # r = p_ite.prepare_generalized()
    # l = p_ite.lexer.tokenize(inp)
    # next(r)
    # for t in l:
    #     print('feeding: ', t)
    #     res = r.send(t)
    #     print(res)
    # res = r.send(None)
    # print(res)

    # t = p_ite.parse_generalized(inp)
