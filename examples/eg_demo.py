import preamble
from metaparse import LALR

# Global context/environment for language semantics.
context = {}

class pCalc(metaclass=LALR.meta):

    "A language for calculating expressions."

    # ===== Lexical patterns / Terminals =====
    # - Patterns specified with regular expressions
    # - Patterns will be tested in declaration order during tokenizing

    IGNORED = r'\s+'             # Special pattern to be ignored.

    EQ  = r'='
    POW = r'\*\*', 3             # Can specify precedence of token (for LALR conflict resolution)
    POW = r'\^'  , 3             # Alternative patterns can share the same name
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    ID  = r'[_a-zA-Z]\w*'
    NUM = r'[1-9][0-9]*'
    def NUM(value):              # Can specify handler for lexical pattern!
        return int(value)

    # ===== Syntactic/Semantic rules in SDT-style =====

    def assign(ID, EQ, expr):        # May access global context.
        context[ID] = expr
        return expr

    def expr(NUM):                   # May compute result purely.
        return NUM                   # NUM is passed as int due to the handler!

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
print (pCalc.interpret("z = 9 ^ 2"))
# 81

print (context)


tr = pCalc.parse(" w  = 1 + 2 * 3 ** 4 + 5 ")

# pprint(tr)
print (pCalc.lexer)

for token in pCalc.lexer.tokenize(" foo  = 1 + bar * 2"):
    print(token.pos,
          token.end,
          token.symbol,
          repr(token.lexeme),   # (lexeme) is something literal.
          repr(token.value))    # (value) is something computed by handler, if exists.

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
# qCalc = LALR.load('./eg_demo_dump.py', globals())

# Context instance to be accessed by the loaded parser
# context = {}

# qCalc.interpret('foo = 1 + 9')

# print (context)
# {'foo': 10}



# context = {}
# pCalc.interpret("bar = 10 ^ 3")
# # pCalc1.interpret("bar = 99 + 1")
# print(context)
