class PExpr(object):
    def __init__(self, *ops):
        self.ops = ops
    def __and__(self, other):
        return And(self, other)
    def __or__(self, other):
        return Or(self, other)
    def __rshift__(self, func):
        return Seman(self, func)

class Token(PExpr):
    def __call__(self, inp):
        lit, = self.ops
        if inp.startswith(lit):
            yield lit, inp[len(lit):]

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

class Seman(PExpr):
    def __call__(self, inp):
        psr, func = self.ops
        for r, inp1 in psr(inp):
            yield func(*r), inp1

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

a = Token('a')
b = Token('b')

assert(list(a('abc'))) == [('a', 'bc')]
assert(list(b('abc'))) == []
assert(list(b('bbc'))) == [('b', 'bc')]

c = Token('c')
ab = a & b
abc = a & b & c

assert(list(ab('abc')))  == [(('a', 'b'), 'c')]
assert(list(abc('abc'))) == [(('a', 'b', 'c'), '')]
assert(list(abc('abd'))) == []

a_b = a | b
assert(list(a_b('abc'))) == [('a', 'bc')] 
assert(list(a_b('bbc'))) == [('b', 'bc')]
assert(list(a_b('cbc'))) == []

assert(list(Many(a)('b'))) == [([], 'b')]
assert(list(Many(a_b)('aaaab'))) == [(['a', 'a', 'a', 'a', 'b'], '')]


# Application

White = Many(Token(' ') | Token('\t') | Token('\n') | Token('\v'))

assert(list(White('   \n  \v b'))) == [([' ', ' ', ' ', '\n', ' ', ' ', '\v', ' '], 'b')]

class Word(PExpr):
    def __init__(self, lit):
        self.lit = lit
    def __call__(self, inp):
        for (w, s), inp1 in (Token(self.lit) & White)(inp):
            yield w, inp1


assert(list(Word('abc')('abc   de'))) == [('abc', 'de')]


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
             a & S & (a >> str.upper))
    return S

pprint([*A('a a a   a a    a ')])


@run
def sexp():
    # sexp = rule(
    #     lambda:
    #     (Word('a') | Word('b')) |
    #     Word('(') & Many(sexp) & Word(')')
    # )
    sexp = rule(lambda:
                (Many1(Token('a') | Token('b')) & White) >> (lambda lst, w: ''.join(lst)) |
                (Word('(') & Many(sexp) & Word(')')) >> (lambda l, e, r: e)
    )
    return sexp

pprint([*sexp('(  (a)  b (bb a) ab)')])

assert 0


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

pprint([*sexp('(  (a)  b (b a) b)')])
