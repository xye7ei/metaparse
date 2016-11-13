import preamble
from metaparse import LALR

# Global context/environment for language semantics.
context = {}

class pCalc(metaclass=LALR.meta):

    "A language for calculating expressions."

    # ===== Lexical patterns / Terminals =====
    # - Will be matched in order when tokenizing

    IGNORED = r'\s+'             # Special token.

    EQ  = r'='
    NUM = r'[1-9][0-9]*'
    ID  = r'[_a-zA-Z]\w*'
    POW = r'\*\*', 3             # Can specify precedence of token (mainly for LALR)
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    # ===== Syntactic/Semantic rules in SDT-style =====

    def assign(ID, EQ, expr):        # May rely on side-effect...
        context[ID] = expr
        return expr

    def expr(NUM):                   # or return local results for purity
        return int(NUM)

    def expr(ID):
        return context[ID]

    def expr(expr_1, ADD, expr_2):   # With TeX-subscripts, meaning (expr → expr₁ + expr₂)
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):     # Can ignore one of the subscripts.
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1


from pprint import pprint

print (type(pCalc))

print (pCalc.interpret("x = 1 + 4 * 3 ** 2 + 5"))
# 42
print (pCalc.interpret("y = 5 + x * 2")) # Here `x` is extracted from the context `context`
# 89
print (pCalc.interpret("z = 99"))
# 99

print (context)


tr = pCalc.parse(" w  = 1 + 2 * 3 ** 4 + 5 ")

# pprint(tr)

for token in pCalc.lexer.tokenize(" w  = 1 + x * 2"):
    print(token.pos,
          token.end,
          token.symbol,
          repr(token.lexeme))

# 1 2 ID 'w'
# 4 5 EQ '='
# 6 7 NUM '1'
# 8 9 ADD '+'
# 10 11 ID 'x'
# 12 13 MUL '*'
# 14 15 NUM '2'

('assign',
 [('ID', 'w'),
  ('EQ', '='),
  ('expr',
   [('expr',
     [('expr', [('NUM', '1')]),
      ('ADD', '+'),
      ('expr',
       [('expr', [('NUM', '2')]),
        ('MUL', '*'),
        ('expr',
         [('expr', [('NUM', '3')]),
          ('POW', '**'),
          ('expr', [('NUM', '4')])])])]),
    ('ADD', '+'),
    ('expr', [('NUM', '5')])])])


# s = pCalc.dumps()
# print(s)
# pCalc.dump('./eg_demo_dump.py')


# Let loaded parser be able to access current runtime env `globals()`.
qCalc = LALR.load('./eg_demo_dump.py', globals())

# Context instance to be accessed by the loaded parser
context = {}

qCalc.interpret('foo = 1 + 9')

print (context)
# {'foo': 10}


@LALR.verbose
def pCalc(lex, rule):           # Parameter (lex, rule) is required by the decorator!

    lex(IGNORED = r'\s+')
    lex(NUM = r'[0-9]+')
    lex(EQ  = r'=')
    lex(ID  = r'[_a-zA-Z]\w*')
    lex(POW = r'\*\*', p=3)
    lex(MUL = r'\*'  , p=2)
    lex(ADD = r'\+'  , p=1)

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


context = {}
pCalc.interpret("bar = 99 + 1")
print(context)
