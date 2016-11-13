# -*- coding: utf-8 -*-

import preamble

from metaparse import LALR

# Global stuff
table = {}

@LALR.verbose
def calc(lex, rule):

    lex(IGNORED = r'\s+')

    @lex(NUM = r'[0-9]+')
    def NUM(val):
        return int(val)

    lex(LEFT = r'\(')
    lex(RIGHT = r'\)')

    lex(EQ  = r'=')
    lex(ID  = r'[_a-zA-Z]\w*')

    lex(POW = r'\*\*', p = 3)
    lex(MUL = r'\*', p = 2)
    lex(ADD = r'\+', p = 1)
    lex(SUB = r'\-', p = 1)

    @rule
    def stmt(assign):
        return assign
    @rule
    def stmt(expr):
        return expr
    
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
    def expr(LEFT, expr, RIGHT):
        return expr

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

table = {}

calc.interpret('x  =  8')
calc.interpret('y  =  x -  6 ')
calc.interpret('z  =  x ** y ')

calc.interpret(' (3) ')
calc.interpret(' x = 03 ')
calc.interpret(' y = 4 * x ** (2 + 1) * 2')

print(table)

# print(calc.dumps())
calc1 = LALR.loads(calc.dumps(), globals())

calc1.interpret(' w = x + 1')

print(table)
