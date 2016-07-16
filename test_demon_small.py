# -*- coding: utf-8 -*-

import pprint as pp
from metaparse import LALR

# Global stuff
table = {}

# class Calc(metaclass=LALR.meta):

#     # ===== Lexical patterns / Terminals =====
#     # - Will be matched in order when tokenizing

#     IGNORED = r'\s+'             # Special token.

#     EQ  = r'='
#     NUM = r'[0-9]+'
#     ID  = r'[_a-zA-Z]\w*'
#     POW = r'\*\*', 3            # Can specify precedence of token (mainly for LALR)
#     MUL = r'\*'  , 2
#     ADD = r'\+'  , 1

#     # ===== Syntactic/Semantic rules in SDT-style =====

#     def assign(ID, EQ, expr):        # May rely on side-effect...
#         table[ID] = expr

#     def expr(NUM):                   # or return local results for purity
#         return int(NUM)

#     def expr(expr_1, ADD, expr_2):   # With TeX-subscripts, meaning (expr → expr₁ + expr₂)
#         return expr_1 + expr_2

#     def expr(expr, MUL, expr_1):     # Can ignore one of the subscripts.
#         return expr * expr_1

#     def expr(expr, POW, expr_1):
#         return expr ** expr_1


from metaparse import cfg, cfg2, rule, v2

# @LALR
# @cfg2
# class Calc1(object):

@v2
def Calc1():

    """Some interesting grammar.
    """

    IGNORED = r'\s+'

    EQ  = r'='
    NUM = r'[0-9]+'
    ID  = r'[_a-zA-Z]\w*'
    POW = r'\*\*', 3
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    def assign(ID, EQ, expr):
        table[ID] = expr

    def expr(NUM):
        return int(NUM)

    def expr(expr_1, ADD, expr_2):
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1

    return


import ast
import inspect
import sys


# print(e[-1]('abc', None, 8))
# print(table)

pp.pprint(Calc1.terminals)
pp.pprint(Calc1)
# Calc1 = cfg2(Calc1)
pCalc1 = LALR(Calc1)
pCalc1.interpret("x = 1 + 2 * 3 ** 4 + 5")
print(table)
# Calc.interpret("x = 1 + 2 * 3 ** 4 + 5")
# Calc.interpret("y = 3 ** 4 * 5")
# Calc.interpret("z = 99")

# print(table)

# tr = Calc.parse(" w  = 1 + 2 * 3 ** 4 + 5 ")
# pp.pprint(Calc1)
# pp.pprint(Calc1.terminals)
# tr = Calc1.parse(" w  = 1 + 2 * 3 ** 4 + 5 ")

# pp.pprint(tr)
