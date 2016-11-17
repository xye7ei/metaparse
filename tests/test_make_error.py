import preamble; from metaparse import *

try:

    class pExpr(metaclass=LALR.meta):

        NUM = '\d+'
        PLUS = '\+'
        TIMES = '\*'

        def expr(expr, PLUS, term):
            return expr + term

        def expr(expr, TIMES, term):
            return expr * term

        def expr(term):
            return term

        def term(NUM):
            return int(NUM)

        def factor(NUM):
            return int(NUM)

    assert 0

except LanguageError as e:
    
    pass


import preamble
from metaparse import *

# Undeclared symbol
try:
    p = LALR()
    with p as (lex, rule):
        lex(a = 'a')
        lex(b = 'b')
        @rule
        def S(a, S, b): pass
        @rule
        def S(): pass
        @rule
        def S(c): pass
    assert 0
except LanguageError as e:
    del p
    # print('Error catched: ', e)


# Unreachable symbol
try:
    p = LALR()
    with p as (l, r):
        l(a = 'a')
        l(b = 'b')
        @r
        def S(a, S, b): pass
        @r
        def S(): pass
        @r
        def B(a): pass
        @r
        def B(b): pass

    assert 0
    t = p.parse('  a   a b  b ')
    v = p.interpret('  a   a b  b ')
    print(t)

except LanguageError as e:
    del p
    # print('Error catched: ', e)

