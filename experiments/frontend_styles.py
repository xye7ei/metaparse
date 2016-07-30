import preamble

from metaparse import *

table = {}

# Clean style
class G_Calc(metaclass=cfg):

    IGNORED = r'\s+'

    EQ  = r'='
    NUM = r'[0-9]+'
    ID  = r'[_a-zA-Z]\w*'
    POW = r'\*\*', 3
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    # ERROR handler?

    def assign(ID, EQ, expr):
        table[ID] = expr

    def expr(NUM):
        return int(NUM)

    def expr(ID):
        return table[ID]

    def expr(expr_1, ADD, expr_2):
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1


# Handler style
class G_Calc():

    def IGNORED(lex: r'\v'):
        pass
    def IGNORED(lex: r'\\'):
        pass

    def ERROR(lex: r'\t'):
        print('ERROR')

    def UNRECOGNIZED(lex: r'.'):
        pass

    # Terminals
    def NUM(lex: r'\d+'):
        return int(lex)

    def ID(lex: r'[_a-zA-Z]\w*'):
        return lex

    def L(lex: r'\('):
        return lex
    def R(lex: r'\)'):
        return lex

    L2 = r'\['
    R2 = r'\]'

    def PLUS(lex: r'\+') -> 1:
        return lex
    def POW(lex: r'\*\*') -> 3:
        return lex
    def TIMES(lex: r'\*') -> 2:
        return lex
    
    # Nonterminals
    def assign(ID: r'[_a-zA-Z]\w*',
               EQ: '=',
               expr):
        table[ID] = expr

    def expr(NUM):
        return NUM

    def expr(expr_1, ADD: r'\+', expr_2):
        return expr_1 + expr_2


# Decorator style
def lex(pat, p=0):
    def _(func):
        return (func.__name__, pat, p, func)
    return _
        
class G_Calc():

    @lex(r'\s+', 3)
    def IGNORED(val):
        pass

    @lex(r'\t', 2)
    def ERROR(val):
        print('ERROR!')

