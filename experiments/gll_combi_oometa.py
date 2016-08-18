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


# def parser(expr):
#     def f(hdl):
#         def _(inp):
#             for r, inp1 in expr(inp):
#                 if type(expr) == And:
#                     yield hdl(*r), inp1
#                 else:
#                     yield hdl(r), inp1
#         return _
#     return f

# @parser(a & Many(b) & c)
# def p_abc(a, bs, c):
#     return list(map(str.upper, [a, *bs, c]))

# r = list(p_abc('abbbcdef'))
# print(r)


class LazObj(PExpr):
    def __init__(self, name, grammar):
        self.name = name
        self.grammar = grammar
    def __repr__(self):
        return '-{}-'.format(self.name)
    def __call__(self, *args):
        return self.grammar[self.name](*args)

__all__ = {
    P.__name__: P
    for P in [Token, Word, Many, Many1, Opt]
}

from collections import OrderedDict as odict

class Reader(object):
    def __init__(self, context):
        self.context = context
        self.grammar = odict()
    def __setitem__(self, k, v):
        if not k.startswith('__'):
            self.grammar[k] = v
    def __getitem__(self, k):
        if k in self.context:
            return self.context[k]
        else:
            if k not in self.grammar:
                self.grammar[k] = LazObj(k, self.grammar)
            return self.grammar[k]

class GLL(object):
    def __init__(self, grammar):
        for k, v in grammar.items():
            setattr(self, k, v)
    def __repr__(self):
        return repr(self.__dict__)
        

class gll(type):
    @classmethod
    def __prepare__(mcls, n, bs, **kw):
        return Reader(kw['context'])
    def __new__(mcls, n, bs, rd, **kw):
        return GLL(rd.grammar)


class G(metaclass=gll, context=__all__):
    
    expr = term | term & Many1(Word('+') & term)
    term = factor | factor & Many1(Word('*') & factor)
    factor = number | Word('(') & expr & Word(')')
    number = Many1(Token('0') | Token('1'))

# print(G)
print()
pprint([*Many1(Token('1'))('0')])
pprint([*Many1(Token('1'))('10')])
pprint([*Many1(Token('1'))('11110')])

print()
from pprint import pprint
pprint([*G.expr('110*(0+10)')])

class A(metaclass=gll, context=__all__):
    a = Word('a')
    S = a & a | a & S & a

print()
pprint([*A.S('a   ')])
# pprint([*(Word('a') & Word('a'))('a  a  ')])
# pprint([*A.A('a   ')])
pprint([*A.S('a   a  ')])
# pprint([*A.S('aaa')])
# pprint([*A.S('aaaa')])
# pprint([*A.S('aaaaa')])
pprint([*A.S('aaaaaa')])
# pprint([*A.S('aaaaaaa')])
# pprint([*A.S('aaaaaaaa')])


def inp(lz):
    def f(i):
        yield from lz()(i)
    return f




def A():

    a = Word('a')

    # @inp
    # def S():
    #     return a & a | a & S & a

    # S = inp(lambda: a & a | a & S & a)

    S = lambda: a & a | a & S & a
    S = inp(S)

    return S

A = A()

pprint([*A('a a a   a a    a ')])





class Grammar(dict):
    def __getattr__(self, k):
        if not k in self:
            return LazObj(k, self)
        else:
            return self[k]
    def __setattr__(self, k, v):
        self[k] = v


def run(func):
    return func(Grammar())

@run
def S(n):
    a = Word('a')
    n.S = a & a | a & n.S & a
    return n.S

pprint([*S('a a a a a a')])
