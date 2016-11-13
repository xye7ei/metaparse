import preamble
from metaparse import LALR, GLR

class pIfThenElse(metaclass=GLR.meta):

    IF     = r'if'
    THEN   = r'then'
    ELSE   = r'else'
    EXPR   = r'\d+'
    SINGLE = r'[_a-zA-Z]+'

    def stmt(ifstmt):
        return ifstmt 

    def stmt(SINGLE):
        return SINGLE 

    def ifstmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return ('ite', EXPR, stmt_1, stmt_2) 

    def ifstmt(IF, EXPR, THEN, stmt):
        return ('it', EXPR, stmt)

from pprint import pprint

res = pIfThenElse.interpret_generalized('if 1 then if 2 then if 3 then a else b else c')
pprint(res)



class pIfThenElse(metaclass=LALR.meta):

    IF     = r'if'
    THEN   = r'then', 1
    ELSE   = r'else', 2
    EXPR   = r'\d+'
    SINGLE = r'[_a-zA-Z]+'

    def stmt(ifstmt):
        return ifstmt 

    def stmt(SINGLE):
        return SINGLE 

    def ifstmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return ('ite', EXPR, stmt_1, stmt_2) 

    def ifstmt(IF, EXPR, THEN, stmt):
        return ('it', EXPR, stmt)

res = pIfThenElse.interpret('if 1 then if 2 then if 3 then a else b else c')
pprint(res)
