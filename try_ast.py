import ast, inspect

from metaparse import grammar
from metaparse import WLL1, LALR

class SExp(object):

    symbs = []


@grammar
def sexp():

    IGNORED = r'\s+'
    L1 = r'\('
    R1 = r'\)'
    SYMBOL = r'[^\[\]\(\)\{\}\s]+'

    def sexp(SYMBOL):
        SExp.symbs.append(SYMBOL)
        return
    def sexp(L1, slist, R1):
        return

    def slist():
        return
    def slist(sexp, slist): 
        return

# def c():
#     i = 0
#     def d():
#         nonlocal i
#         i += 1
#         print(i)
#     return d

#     return slist                # For inspection.
# g_sexp = grammar(sexp)

# co = inspect.getsource(sexp)
# md = ast.parse(co)
# fd = md.body[0]
# type(fd)
# fd.__dict__
# sfds = [sfd for sfd in fd.body if isinstance(sfd, ast.FunctionDef)]
# sfds
# sfd3 = sfds[3]
# sfd3.__dict__

p_sexp = WLL1(sexp)
# parse = SExp.sexp.parse
# interpret = SExp.sexp.interpret

t = p_sexp.interpret('(   ( a b  )  (c)  (((d)) e))')

print(SExp.symbs)

# t = parse('(   ( a b  )  (c)  (((d)) e))')
# t.interpret('(   ( a b  )  (c)  (((d)) e))')
# print(SExp.N)

# print(sexp)
# import pprint as pp
# pp.pprint(t)

# print(N)
