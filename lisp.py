import pprint as pp
from earley import earley
from lalr import lalr

# class GLisp(metaclass=earley):
class GLisp(metaclass=lalr):

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

# list(GLisp.tokenize(jnp))
# list(enumerate(GLisp.Ks))

# list(enumerate(GLisp.ACTION))
# list(enumerate(GLisp.GOTO))

# GLisp.ACTION
# GLisp.parse(inp)
# ('term',
#  ('appl',
#   ['(',
#    ('term', ('atom', '+')),
#    ('slist',
#     [('term', ('atom', '1')),
#      ('slist',
#       [('term', ('atom', '2')),
#        ('slist',
#         [('term',
#           ('appl',
#            ['(',
#             ('term', ('atom', '*')),
#             ('slist',
#              [('term', ('atom', '3')),
#               ('slist', [('term', ('atom', '4')), 'slist'])]),
#             ')'])),
#          'slist'])])]),
#    ')']))

pp.pprint(GLisp.parse(inp))
