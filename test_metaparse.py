import pprint as pp

from metaparse import *


class GIfThenElse(metaclass=Earley.meta):
    """Ambigious grammar with Dangling-Else structure. """

    IGNORED = r'\s'
    IF      = r'if'
    THEN    = r'then'
    ELSE    = r'else'
    EXPR    = r'\d+'
    SINGLE  = r'[_a-zA-Z]\w*'

    def stmt(SINGLE):
        return SINGLE
    def stmt(IF, EXPR, THEN, stmt):
        return ('it', stmt)
    def stmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return ('ite', stmt_1, stmt_2)


class S(metaclass=Earley.meta):
    """A special ambigious grammar partitioning x sequence,
    which demonstrates Jay Earley's error figuring out the
    method of constructing parse forest correctly, noted by
    Tomita."""
    IGNORED = r'\s+'
    x = r'x'
    def S(S_1, S_2):
        return [S_1, S_2]
    def S(x):
        return '!'


class A(metaclass=Earley.meta):
    """Ambigious grammar with more-to-one completion. Shared completee
    item leads to forking of parent items' stacks.

    """
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

# Ambiguous parsing
assert len(S.interpret('x  x   x')) == 2
assert len(A.interpret('a    a  ')) == 6
assert len(GIfThenElse.interpret('if 1 then if 71 then if 23 then if 987 then aa else bb')) == 4


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
    def num(ZERO):
        return 0
    def num(SUCC, num):
        return 1 + num


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
# li = 'a bc def g'

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

