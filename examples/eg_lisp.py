import preamble

from metaparse import *


class ListParser(metaclass=LALR.meta):
    """A tiny grammar for lists."""
    IGNORED = r'\s'
    SYMBOL  = r'\w+'
    def list(list, SYMBOL):
        list.append(SYMBOL)
        return list
    def list():
        return []


class LISP(metaclass=cfg):
    """A parser for scheme-like grammar. Should be easy to describe and
    parse.

    """

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

    def parlist(SYMBOL, parlist):
        return [SYMBOL] + parlist
    # def parlist(parlist, SYMBOL):
    #     return parlist + [SYMBOL]
    def parlist():
        return []

    def sexps(sexp, sexps):
        return [sexp] + sexps
    # def sexps(sexp, sexps):
    #     return sexps + [sexp]
    def sexps():
        return []

p_lisp = WLL1(LISP)
res = p_lisp.interpret('(lambda (x y) (+ x y))')
print(res)
