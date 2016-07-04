import pprint as pp

from metaparse import *


class GIfThenElse(metaclass=Earley.meta):
    """Ambigious grammar with Dangling-Else. """

    IGNORED = r'\s'
    IF      = r'if*'
    THEN    = r'then*'
    ELSE    = r'else*'
    EXPR    = r'\(\s*e\s*\)'
    SINGLE  = r's'

    def stmt(SINGLE):
        return 'sg'
    def stmt(IF, EXPR, THEN, stmt):
        return ['IF', 'x', 'THEN', stmt]
    def stmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return ['IF', 'x', 'THEN', stmt_1, 'ELSE', stmt_2]


class S(metaclass=Earley.meta):
    """A special ambigious grammar partitioning x sequence."""
    IGNORED = r'\s+'
    x = r'x'
    def S(S_1, S_2):
        return S_1 + S_2
    def S(x):
        return ['x']


class A(metaclass=Earley.meta):
    """Ambigious grammar with more-to-one completion."""
    IGNORED = r'\s+'
    a = r'a'
    def A(B, C): return

    def B(D):    return
    def B(E):    return

    def C(F):    return
    def C(G):    return
    def C(H):    return

    def D(a):    return
    def E(a):    return
    def F(a):    return
    def G(a):    return
    def H(a):    return


# PHASE 1: Test Earley

pp.pprint(S.parse_forest('x  x   x'))
pp.pprint(A.parse_forest('a a'))

# ParserC.parse_states('a b')
# pp.pprint(ParserC.states)
# res = ParserC.parse_forest('a b')
# pp.pprint(res)
# pp.pprint(ParserC.forest)

# pp.pprint(list(ParserC.grammar.tokenize('a b', False)))
# ParserC.parse_states('a b')
# pp.pprint(ParserC.states)
# pp.pprint(ParserC.forest)
# ParserC.parse_chart('a b')
# pp.pprint(ParserC.chart)

# print()
# res = GIfThenElse.parse_forest('if (e) then if (e) then if (e) then s else s')
# print(len(res))
# pp.pprint(res)
# pp.pprint(GIfThenElse.forest[-1])
# GIfThenElse.parse_chart('if (e) then if (e) then s else s')
# pp.pprint(GIfThenElse.states)
# pp.pprint(GIfThenElse.chart)
# trs = GIfThenElse.find_trees('stmt', 0)
# print(len(trs))

# trs0 = [] 
# for tr in trs:
#     if tr not in trs0:
#         trs0.append(tr)
# print(len(trs0))

# pp.pprint(GIfThenElse.find_trees('stmt', 0, 9))
# print(len(GIfThenElse.find_trees('stmt^',0))) # WHY LENGTH==4????
# pp.pprint(GIfThenElse.find_nodes(0, 9, 'stmt^'))
# GIfThenElse.parse('if (e) then if (e) then s else s')



# ParserC.parse('a    b  b')
# pp.pprint(ParserC.chart)


# PHASE 2: Test LALR

class ListParser(metaclass=LALR.meta):
    IGNORED = r'\s'
    SYMBOL  = r'\w+'
    def list(list, SYMBOL):
        list.append(SYMBOL)
        return list
    def list():
        return []


class ChurchParser(metaclass=LALR.meta):
    SUCC = 'succ'
    ZERO = 'zero'
    def num(ZERO): return 0
    def num(SUCC, num): return 1 + num


class SExpParser(metaclass=LALR.meta):
    """A parser for scheme-like grammar."""

    LAMBDA = r'\(\s*lambda'
    LEFT   = r'\('
    RIGHT  = r'\)'
    SYMBOL = r'[^\(\)\s]+'

    # _env = {}
    # def _unify():
    #     pass

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

    # def sexps(sexp, sexps):
    #     return [sexp] + sexps
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
# res = SExpParser.parse('(lambda (x y) (+ x y))')
# pp.pprint(res)
# pp.pprint(res.translate())
# pp.pprint(SExpParser.interpret('(lambda (x y) (+ x y))'))
