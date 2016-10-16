import lessparse as lp


p = lp.LALR()

p.lexer.word(
    IGNORED=' ',
    PLUS='+',
    TIMES='*',
    LEFT='(',
    RIGHT=')'
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
pprint(p.grammar)
# p.inspect_ACTION
# t = p.parse('123')
# pprint(t)
tkns = (p.lexer.tokenize('123 + 8'))
q = p.prepare()
next(q)
q.send(next(tkns))
q.send(next(tkns))
q.send(next(tkns))
t = q.send(None)
assert t == 131

t = p.parse('123 + 8')
assert p.interpret('123 + 8') == 131
t = p.parse('123 + 2 * 1')
assert p.interpret('123 + 2 * 1') == 125
assert p.interpret('123 + 2 * (1 + 2)') == 129

tough = ' + '.join(['(2 * (1 + 1) + 2 * 2)'] * 1000)


# %timeit p.interpret(tough)
# 1 loops, best of 3: 346 ms per loop
