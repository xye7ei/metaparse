import preamble
from metaparse import LALR

pCalc = LALR()

lex  = pCalc.lexer
rule = pCalc.rule

# lex(<terminal-symbol> = <pattern>)
lex(IGNORED = r'\s+')
lex(NUM = r'[0-9]+')
lex(EQ  = r'=')
lex(ID  = r'[_a-zA-Z]\w*')

# lex(... , p = <precedence>)
lex(POW = r'\*\*', p=3)
lex(POW = r'\^')                # No need to give the precedence twice for POW.
lex(MUL = r'\*'  , p=2)
lex(ADD = r'\+'  , p=1)

# @rule
# def <lhs> ( <rhs> ):
#     <semantics>
@rule
def assign(ID, EQ, expr):
    context[ID] = expr
    return expr

@rule
def expr(ID):
    return context[ID]

@rule
def expr(NUM):
    return int(NUM)

@rule
def expr(expr_1, ADD, expr_2):
    return expr_1 + expr_2

@rule
def expr(expr, MUL, expr_1):
    return expr * expr_1

@rule
def expr(expr, POW, expr_1):
    return expr ** expr_1

# Complete making the parser after collecting things!
pCalc.make()

context = {}
pCalc.interpret("x = 3")
pCalc.interpret("y = x ^ 2")
pCalc.interpret("z = x + y + 1")

from pprint import pprint
print(context)
print(pCalc.precedence)
