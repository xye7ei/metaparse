import preamble
import metaparse
from metaparse import LALR


class ListParser(metaclass=LALR.meta):
    """A tiny grammar for lists."""
    IGNORED = r'\s'
    SYMBOL  = r'\w+'
    def list(list, SYMBOL):
        list.append(SYMBOL)
        return list
    def list():
        return []


class LISP(metaclass=LALR.meta):
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

    def sexps(sexps, sexp):
        return sexps + [sexp]
    # def sexps(sexp, sexps):
    #     return sexps + [sexp]
    def sexps():
        return []


p_lisp = (LISP)

lx = p_lisp.lexer
p = p_lisp.prepare(False)
next(p)

inp = '(+ (+ 1 2) 3 ))'
tks = list(lx.tokenize(inp, True))


from pprint import pprint

# pprint(tks)

for tk in tks:
    opt = p.send(tk)
    if isinstance(opt, metaparse.ParseError):
        pprint(opt.args)
        pprint(opt.__dict__)
    else:
        pprint(opt.result)

res = p_lisp.interpret('(lambda (x y) (+ x y) ))')
print(res)

# for tk in tks:
#     res = p.send(tk).result
#     pprint(res)
