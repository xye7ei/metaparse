import pprint as pp

from metaparse import cfg, Earley, LALR, lalr

@Earley
class GIfThenElse(metaclass=cfg):
    # IGNORED = r'\s'
    IF      = r'if\s*'
    THEN    = r'then\s*'
    ELSE    = r'else\s*'
    EXPR    = r'\(\s*e\s*\)\s*'
    SINGLE  = r's'

    def stmt(SINGLE):
        return 'sg'
    def stmt(IF, EXPR, THEN, stmt):
        return ['IF', 'x', 'THEN', stmt]
    def stmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return ['IF', 'x', 'THEN', stmt_1, 'ELSE', stmt_2]

@Earley
class ParserS(metaclass=cfg):
    IGNORED = r'\s'
    x = r'x'
    def S(S_1, S_2):
        return S_1 + S_2
    def S(x):
        return ['x']

@Earley
class ParserC(metaclass=cfg):
    IGNORED = r'\s'
    a = r'a'
    b = r'b'
    def C(a, E, F): return
    def C(a, F, E): return
    def E(b): return
    def F(b): return


# PHASE 1: Test Earley

# print()
GIfThenElse.parse_chart('if (e) then if (e) then s else s')
pp.pprint(GIfThenElse.chart) 
# pp.pprint(GIfThenElse.chart)

# ParserC.parse('a    b  b')
# pp.pprint(ParserC.chart)
# pp.pprint(ParserC.graph)
# pp.pprint(ParserC.edges)


# PHASE 2: Test LALR

@LALR
class ListParser(metaclass=cfg):
    IGNORED = r'\s'
    SYMBOL  = r'\w+'
    def list(list, SYMBOL):
        list.append(SYMBOL)
        return list
    def list():
        return []

@LALR
class ChurchParser(metaclass=cfg):
    SUCC = 'succ'
    ZERO = 'zero'
    def num(ZERO): return 0
    def num(SUCC, num): return 1 + num

@LALR
class SExpParser(metaclass=cfg):

    LAMBDA = r'\(\s*lambda'
    LEFT   = r'\('
    RIGHT  = r'\)'
    SYMBOL = r'[^\(\)\s]+'

    def sexp(var):
        return var
    def sexp(abst):
        return abst
    def sexp(appl):
        return appl

    def var(SYMBOL):
        return SYMBOL
    def abst(LAMBDA, LEFT, parlist, RIGHT_1, sexp, RIGHT_2):
        return ('LAMBDA', parlist, sexp)
    def appl(LEFT, sexp, sexps, RIGHT):
        return [sexp, sexps]

    def parlist(parlist, SYMBOL):
        return parlist + [SYMBOL]
    def parlist():
        return []

    def sexps(sexps, sexp):
        return sexps + [sexp]
    def sexps():
        return []

# xx = '<person name="john"></person>'
li = 'a bc def g'

# print()
# pp.pprint(ListParser.parse('x x  x'))
# pp.pprint(ListParser.interpret('x x  x'))

# print()
# pp.pprint(ChurchParser.interpret('succ succ succ succ zero'))


# print()
# pp.pprint(list(SExpParser.grammar.tokenize('(lambda (x y) (+ x y))', True)))
# pp.pprint(SExpParser.parse('(lambda (x y) (+ x y))'))
# pp.pprint(SExpParser.interpret('(lambda (x y) (+ x y))'))
