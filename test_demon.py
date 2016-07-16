from metaparse import LALR

# Global stuff
table = {}
results = []

class Calc(metaclass=LALR.meta):

    # ===== Lexical patterns / Terminals =====
    # - Will be matched in order when tokenizing

    START   = r'\A\s*'
    END     = r'\Z'             # Special token.
    SEPLINE = r'\n\s*'
    IGNORED = '\s+'             # Special token.
    EQ  = r'='
    NUM = r'\d+'
    ID  = r'\w+'
    POW = r'\*\*', 3            # Can specify precedence of token!
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    # ===== Syntactic/Semantic rules in SDT-style =====

    def prog(START, stmts):
        print(table)

    def stmts():
        pass
    def stmts(stmts, stmt):
        results.append(stmt)

    def stmt(ID, EQ, expr, SEPLINE):
        table[ID] = expr
        return expr
    def stmt(expr, SEPLINE):
        return expr

    def expr(ID):
        if ID in table:
            return table[ID]
        else:
            raise ValueError('ID refered before binding: {}'.format(ID))

    def expr(NUM):
        return int(NUM) 

    def expr(expr_1, ADD, expr_2):   # With TeX-subscripts, meaning (expr → expr₁ + expr₂)
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):     # Can ignore one of the subscripts.
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1

inp = """
a = 1

b = 3
c = a + 2 * 3 ** b * 5 + 1

"""


Calc.interpret(inp)

assert table['a'] == 1
assert table['c'] == 1 + 2 * 3 ** 3 * 5 + 1

print(Calc.parse("""
3 + 2 ** 2 * 5 + 1
"""))
