# -*- coding: utf-8 -*-

import preamble

from metaparse import cfg, LALR

# Global stuff
table = {}

@cfg.v2
def Calc():

    IGNORED = r'\s+'

    L   = r'\('
    R   = r'\)' 
    EQ  = r'='
    NUM = r'\d+'
    def NUM(lex):
        return int(lex)
    ID  = r'[_a-zA-Z]\w*' 
    POW = r'\*\*', 3
    MUL = r'\*'  , 2
    MUL = r'x'
    ADD = r'\+'  , 1

    def stmt(ID, EQ, expr):
        table[ID] = expr 
        return expr

    def stmt(expr):
        return expr

    def expr(ID):
        if ID in table:
            return table[ID]
        else:
            raise ValueError('ID yet bound: {}'.format(ID))

    def expr(NUM):
        # return int(NUM) 
        return NUM

    def expr(L, expr, R):
        return expr

    def expr(expr_1, ADD, expr_2):   # With TeX-subscripts, meaning (expr → expr₁ + expr₂)
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):     # Can ignore one of the subscripts.
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1


p = calc = LALR(Calc)

# s = calc.dumps()

# # print(s)

# p = LALR.loads(s, globals())

p.interpret(' (3) ')
p.interpret(' x = 03 ')
p.interpret(' y = 4 * x ** (2 + 1) * 2')

print(table)
