import pprint as pp

from metaparse import *


class GIfThenElse(metaclass=Earley.meta):
    """Ambigious grammar with Dangling-Else. """

    IGNORED = r'\s'
    IF      = r'if*'
    THEN    = r'then*'
    ELSE    = r'else*'
    EXPR    = r'\(\s*e\s*\)'
    SINGLE  = r'\w+'

    def stmt(SINGLE):
        return SINGLE
    def stmt(IF, EXPR, THEN, stmt):
        return ('it', stmt)
    def stmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return ('ite', stmt_1, stmt_2)


class S(metaclass=Earley.meta):
    """A special ambigious grammar partitioning x sequence."""
    IGNORED = r'\s+'
    x = r'x'
    def S(S_1, S_2):
        return [S_1, S_2]
    def S(x):
        return '!'


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

pp.pprint(S.recognize('x  x   x'))
# pp.pprint(S.parse('x  x   x'))
pp.pprint(S.interpret('x  x   x'))
# pp.pprint(S.parse_chart('x  x   x'))
# pp.pprint(A.parse_forest('a a'))

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
pp.pprint(GIfThenElse.recognize('if (e) then if (e) then if (e) then aa else bb'))
pp.pprint(GIfThenElse.interpret('if (e) then if (e) then if (e) then aa else bb'))
# pp.pprint(GIfThenElse.parse('if (e) then if (e) then if (e) then s else s'))


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
