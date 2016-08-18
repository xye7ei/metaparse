class PExpr(object):
    def __init__(self, *ops, hdl=None):
        self.ops, self._hdl = ops, hdl
    def __and__(self, other):
        return And(self, other)
    def __or__(self, other):
        return Or(self, other)
    def trans(self, args):
        return self._hdl(args) if self._hdl else args

class Token(PExpr):
    def __init__(self, lit, hdl=None):
        self.lit, self._hdl = lit, hdl
    def __call__(self, inp):
        if inp.startswith(self.lit):
            yield self.trans(self.lit), inp[len(self.lit):]

class And(PExpr):
    def __call__(self, inp):
        lop, rop = self.ops
        for r1, inp1 in lop(inp):
            for r2, inp2 in rop(inp1):
                if type(lop) is And:
                    yield r1 + (r2,), inp2
                else:
                    yield (r1, r2), inp2

class Or(PExpr):
    def __call__(self, inp):
        for psr in self.ops:
            yield from psr(inp)

class Many(PExpr):
    def __call__(self, inp):
        got = False
        for r, inp1 in self.ops[0](inp):
            got = True
            for rs, inp2 in self(inp1):
                yield [r] + rs, inp2
        if not got:
            yield [], inp

class Opt(PExpr):
    def __call__(self, inp):
        got = False
        for r1, inp1 in self.ops[0](inp):
            got = True
            yield ([r1], inp1)
        if not got:
            yield ([], inp)

# 
class Many1(PExpr):
    def __call__(self, inp):
        for r, inp1 in self.ops[0](inp):
            got = False
            for rs, inp2 in self(inp1):
                got = True
                yield [r] + rs, inp2
            if not got:
                yield [r], inp1
                

# class Many(PExpr):
#     def __call__(self, inp):
#         psr, = self.psrs
#         agd = [([], inp)]
#         while 1:
#             agd1 = []
#             for rs, inp in agd:
#                 for r1, inp1 in psr(inp):
#                     agd1.append((rs+[r1], inp1))
#             if agd1: agd = agd1
#             else: break
#         for rs, inp in agd:
#             yield rs, inp

import re

from pprint import pprint
print()

a = Token('a')
b = Token('b')
c = Token('c')

print(list(a('abc')))
print(list(b('abc')))
print(list(b('bbc')))

print()
ab = a & b
abc = a & b & c

print(list(ab('abc')))
print(list(abc('abc')))
print(list(abc('abd')))

print()
a_b = a | b
print(list(a_b('abc')))
print(list(a_b('bbc')))
print(list(a_b('cbc')))

print()
print(list(Many(a)('b')))
print(list(Many(a)('aaaab')))


# Application

White = Many(Token(' ') | Token('\t') | Token('\n') | Token('\v'))

print()
print(list(White('   \n  \v b')))

class Word(PExpr):
    def __init__(self, lit):
        self.lit = lit
    def __call__(self, inp):
        for (w, s), inp1 in (Token(self.lit) & White)(inp):
            yield w, inp1


print(list(Word('abc')('abc   de')))



# =========================
#         Frontend
# =========================

def rule(lz):
    def f(inp):
        yield from lz()(inp)
    return f
def run(func):
    return func()

@run
def A():
    a = Word('a')
    S = rule(lambda:
            a & a |
            a & S & a)
    return S

pprint([*A('a a a   a a    a ')])



class LazExpr(PExpr):
    def __init__(self, name, grammar):
        self.name = name
        self.grammar = grammar
    def __repr__(self):
        return '-{}-'.format(self.name)
    def __call__(self, *args):
        return self.grammar[self.name](*args)

class Grammar(dict):
    def __getattr__(self, k):
        if not k in self:
            return LazExpr(k, self)
        else:
            return self[k]
    def __setattr__(self, k, v):
        self[k] = v

def gll(func):
    return func(Grammar())

@gll
def S(n):
    a = Word('a')
    n.S = a & a | a & n.S & a
    return n.S

pprint([*S('a a a a a a')])


@gll
def sexp(n):
    n.sexp = (Word('a') | Word('b')) |\
             Word('(') & Many(n.sexp) & Word(')')
    return n.sexp

@run
def sexp():
    sexp = rule(lambda:
               (Word('a') | Word('b')) |
               Word('(') & Many(sexp) & Word(')')
    )
    # sexp = rule(lambda:
    #            (Word('a') | Word('b')) >> Func(f1) |
    #            (Word('(') & Many(sexp) & Word(')')) >> Func(f2)
    # )
    return sexp

pprint([*sexp('(  (a)  b (b a) b)')])
