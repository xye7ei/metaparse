from metaparse import LALR

@LALR.verbose
def calc(lex, rule):

    lex(IGNORED = r'\s+')

    @lex(NUM = r'[0-9]+')
    def NUM(val):
        return int(val)

    lex(EQ  = r'=')
    lex(ID  = r'[_a-zA-Z]\w*')

    lex(POW = r'\*\*', p = 3)
    lex(MUL = r'\*', p = 2)
    lex(ADD = r'\+', p = 1)
    lex(SUB = r'\-', p = 1)

    @rule
    def assign(ID, EQ, expr):
        table[ID] = expr
        return expr

    @rule
    def expr(ID):
        return table[ID]

    @rule
    def expr(NUM):
        return int(NUM)

    @rule
    def expr(expr_1, ADD, expr_2):
        return expr_1 + expr_2

    @rule
    def expr(expr_1, SUB, expr_2):
        return expr_1 - expr_2

    @rule
    def expr(expr, MUL, expr_1):
        return expr * expr_1

    @rule
    def expr(expr, POW, expr_1):
        return expr ** expr_1


from pprint import pprint
# pprint(lex)
# pprint(rule)

# 
table = {}

calc.interpret('x  =  8')
calc.interpret('y  =  x -  6 ')
calc.interpret('z  =  x ** y ')


import unittest

class Test(unittest.TestCase):

    def test(self):
        self.assertEqual(table, dict(x=8, y=2, z=64))


if __name__ == '__main__':
    unittest.main()

