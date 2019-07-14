import metaparse as mp
from metaparse import LALR, END_TOKEN

p = LALR()

p.lexer.more(
    IGNORED=' ',
    PLUS='\+',
    TIMES='\*',
    LEFT='\(',
    RIGHT='\)'
)

@p.lexer(NUMBER='\d+')
def _(val):
    return int(val)

@p.rule
def expr(expr, PLUS, term):
    return expr + term

@p.rule
def expr(term):
    return term

@p.rule
def term(term, TIMES, factor):
    return term * factor

@p.rule
def term(factor):
    return factor


with p as (lexer, rule):

    @rule
    def factor(NUMBER):
        return NUMBER

    @rule
    def factor(LEFT, expr, RIGHT):
        return expr

    # exit and make!

# p.make()


from pprint import pprint
# pprint(p.grammar)
# p.inspect_ACTION
# t = p.parse('123')
# pprint(t)
tkns = (p.lexer.tokenize('123 + 8'))
q = p.prepare()
next(q)
q.send(next(tkns))
q.send(next(tkns))
q.send(next(tkns))
t = q.send(END_TOKEN)
assert t == mp.Just(131)

t = p.parse('123 + 8')
assert p.interpret('123 + 8') == 131
t = p.parse('123 + 2 * 1')
assert p.interpret('123 + 2 * 1') == 125
assert p.interpret('123 + 2 * (1 + 2)') == 129

tough = ' + '.join(['(2 * (1 + (1)) + 2 * 2 + (3))'] * 100)
assert p.interpret(tough) == eval(tough)

# if replication is 10000
# %timeit p.interpret(tough)
# 1 loops, best of 3: 346 ms per loop

p_sexp = LALR()

with p_sexp as (lex, rule):

    # # Order???
    # lex.word(
    #     IGNORED=' ',
    #     LEFT='(',
    #     RIGHT=')',
    #     COMMA=',',
    # )
    # lex.re(
    #     NUMBER='\d+(\.\d*)?',
    #     SYMBOL='\w+',
    #     UNKNOWN='%',
    # )
    lex.more(
        IGNORED='%',
        LEFT='\(',
        RIGHT='\)',
        COMMA=',',
    )
    lex(IGNORED='\s+')
    lex(SYMBOL='[_a-zA-Z]\w*')
    lex(UNKNOWN='&')

    @lex(NUMBER='[1-9]\d*(\.\d*)?')
    def _(val):
        return int(val)

    @rule
    def sexp(atom):
        return atom
    @rule
    def sexp(LEFT, slist, RIGHT):
        return slist


    @rule
    def slist():
        return []
    @rule
    def slist(slist, sexp):
        slist.append(sexp)
        return slist

    @rule
    def atom(NUMBER):
        return NUMBER
    @rule
    def atom(SYMBOL):
        return SYMBOL

# p_sexp.inspect_ACTION
# p_sexp.inspect_GOTO

# debug p_sexp.make()

# s = p_sexp.parse('123')
# pprint(s)
# pprint(list(p_sexp.lexer.tokenize('(a b (c d))')))
# pprint(p_sexp.lexer)

# ds = (p_sexp.dumps())
# ctx = {}
# exec(ds, {}, ctx)
# pprint(ctx)

# lx_dp = p_sexp.lexer.dumps()
# print(lx_dp)
# lexer1 = Lexer.loads(lx_dp, globals())

# print(list(lexer1.tokenize(' 123  99 ')))


import warnings

with warnings.catch_warnings(record=True) as w:
    s = p_sexp.interpret('(a 123 (c (d)) %  & e)')
    assert len(w) == 1

assert s == ['a', 123, ['c', ['d']], 'e'], s


sexp_dp = p_sexp.dumps()

with open('sexp_dump.py', 'w') as o:
    o.write(sexp_dp)

# print(sexp_dp)
p_sexp1 = LALR.loads(sexp_dp, globals())

with warnings.catch_warnings(record=True) as w:
    s = p_sexp.interpret('(a & 123 (c (d)) %  & e)')
    assert len(w) == 2
    

assert s == ['a', 123, ['c', ['d']], 'e'], s

