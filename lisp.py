import pprint as pp
from earley import earley, Earley
from grammar import cfg
from lalr import lalr, LALR

class GLisp(metaclass=cfg):

    LAMBDA = r'lambda'
    STRING = r'"[^\"]*"'
    NUMBER = r'[+-]?[1-9]\d*(\.\d+)?([Ee][+-]?\d+)?'
    SYMBOL = r'[^\s\(\)\[\]\{\}]+'
    NIL    = r'NIL'
    LEFT   = r'\('
    RIGHT  = r'\)'

    # LEFT2  = r'\['
    # RIGHT2 = r'\]'
    # LEFT3  = r'\{'
    # RIGHT3 = r'\}'


    def term(atom): return atom

    def term(abst): return abst

    def term(appl): return appl


    def atom(STRING): return STRING

    def atom(NUMBER): return NUMBER

    def atom(SYMBOL): return SYMBOL


    def abst(LEFT, LAMBDA, LEFT1, parlist, RIGHT1, term, RIGHT):
        return {'lambda.' + '.'.join(parlist) : term}


    def parlist(SYMBOL, parlist): return [SYMBOL] + parlist

    def parlist(): return []


    def appl(LEFT, term, slist, RIGHT): return (term, slist)


    def slist(term, slist): return [term] + slist

    def slist(): return []


inp = '(+ 1 2 (* 3 4))'
jnp = '((lambda (x) (+ x 1)) 9)'

pe = Earley(GLisp)
pl = LALR(GLisp)

# pp.pprint(GLisp.parse(inp))
# pp.pprint(pe.parse_process(inp))

# %timeit pe.parse_process(inp)
# %timeit pl.parse(inp)

# %timeit pe.parse_process(jnp)
# %timeit pl.parse(jnp)
