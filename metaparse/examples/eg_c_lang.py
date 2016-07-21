from metaparse import *

# PHASE 1

class Lang(metaclass=cfg):

    """A C-style language with LEFT-SHARING problem."""

    IGNORED = r'\s+'

    L3 = r'\{'
    R3 = r'\}'

    # OP = r'\+-\*\/'
    EQ = r'='

    IF = r'if'
    ELSE = r'else'
    SEMI = r';'

    WORD = r'\w+'

    def prog(stmts):
        return {'prog': stmts}

    def stmts(stmt, stmts):
        return [stmt, *stmts]
    # Whether to allow empty statements
    def stmts():
        return []
    # def stmts(stmt):
    #     return [stmt]

    def stmt(assign, SEMI):
        return assign
    def stmt(WORD):
        return ('expr', WORD)
    def stmt(block):
        return block
    def stmt(ifstmt):
        return ifstmt

    # Left sharing to test GLL parser
    # ? Dangling-Else problem
    # ! GLL should handle this correctly, yielding parse forest!
    # ! Should consider resolve this natural ambiguity
    def ifstmt(IF, WORD, stmt):
        return ('it', WORD, stmt)
    # def ifstmt(IF, WORD, stmt):
    #     return ('it', WORD, stmt)
    def ifstmt(IF, WORD, stmt_1, ELSE, stmt_2):
        return ('ite', WORD, stmt_1, stmt_2)

    # Resolve ambiguity by design!
    # One solution:
    # def stmt(matched_stmt):
    #     return matched_stmt
    # def stmt(open_stmt):
    #     return open_stmt
    # def matched_stmt(IF, WORD, matched_stmt_1, ELSE, matched_stmt_2):
    #     return ('ite', WORD, matched_stmt_1, matched_stmt_2)
    # def matched_stmt(other):
    #     return other
    # def open_stmt(IF, WORD, stmt):
    #     return ('it', WORD, stmt)
    # def open_stmt(IF, WORD, matched_stmt, ELSE, open_stmt):
    #     return ('it', matched_stmt, open_stmt)

    def block(L3, stmts, R3):
        return stmts

    def assign(WORD_1, EQ, WORD_2):
        return ('assign', WORD_1, WORD_2)


inp = """
if abc {

a = b;

} else {

c

}
"""

inp = """
if x if y if z a = b; else c = d; else e = f;
"""

psr = Earley(Lang)              # Earley needs to handle ε-Rules

# print('Recognization table:')
# rec = psr.recognize(inp)
# pp.pprint(rec)

# print('Chart:')
# cht = psr.parse_chart(inp)
# pp.pprint(cht)

print('Parse trees:')
res = psr.parse(inp)
its = psr.interpret(inp)
pp.pprint(its)
# print(len(res), ' trees generated.')

# print('Interpret results:')
# pp.pprint(psr.interpret(inp))
