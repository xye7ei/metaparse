import preamble
import ast

from metaparse import *

# class G_Calc(metaclass=cfg):
# @cfg.v2
def G_Calc():

    IGNORED = r'\s+'

    EQ  = r'='
    NUM = r'[1-9]\d*'
    def NUM(lex):
        return float(lex)

    ID  = r'[_a-zA-Z]\w*'
    POW = r'\*\*', 3
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    def assign(ID, EQ, expr):
        table[ID] = expr

    def expr(NUM):
        return NUM

    def expr(ID):
        return table[ID]

    def expr(expr_1, ADD, expr_2):
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1

# assert 0

G_Calc = grammar(G_Calc)

from pprint import pprint

p = LALR(G_Calc)

# with open('eg_dumps_file.py', 'w') as o:
#     psr_fl = p.dumps()
#     o.write(psr_fl)

# with open('eg_dumps_file.py', 'r') as o:
#     s = o.read()
#     p = LALR.loads(s, globals())

p.dump('eg_dumps_file.py')
p.load('eg_dumps_file.py', globals())

pprint(p.__dict__)
# pprint(ctx)

# timeit LALR.loads(s, globals())
# timeit p = LALR(G_Calc)

s1 = p.dumps()
p1 = LALR.loads(s1, globals())
s2 = p1.dumps()
p2 = LALR.loads(s2, globals())

table = {}
p2.interpret('x = 3')
p2.interpret('y = x ** 2 * 2 + 1')
pprint(table)
