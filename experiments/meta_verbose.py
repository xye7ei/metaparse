import preamble
import ast

from metaparse import Grammar, Rule, LALR

from textwrap import dedent
from inspect import getsource

import inspect
import textwrap

class Logger:

    lex_names = []
    lex_pats = []
    lex_srcs = {}
    lex_hdls = {}
    lex_preces = {}
    rules = []
    semans = []
    rule_srcs = []


def G(cls):
    d = {}
    for k, v in Logger.__dict__.items():
        if not k.startswith('__'):
            d[k] = v
    g = Grammar(
        Logger.lex_names, Logger.lex_pats,
        Logger.rules, Logger.semans,
        Logger.lex_preces,
        Logger.lex_hdls)
    # clean up
    Logger.lex_names = []
    Logger.lex_pats = []
    Logger.lex_hdls = {}
    Logger.lex_srcs = {}
    Logger.lex_preces = {}
    Logger.rules = []
    Logger.semans = []
    Logger.rule_srcs = []
    return g

def lex_hdl(name):
    def _(func):
        src = dedent(getsource(func))
        # Logger.lex_srcs[name] = src[src.index('def'):]
        Logger.lex_srcs[name] = src
        Logger.lex_hdls[name] = func
        return func
    return _

def lex(*p, **kw):
    p = -1
    if 'p' in kw:
        p = kw.pop('p')
    name, pat = kw.popitem()
    Logger.lex_names.append(name)
    Logger.lex_pats.append(pat)
    Logger.lex_preces[name] = p
    return lex_hdl(name)

def rule(func):
    Logger.rules.append(Rule.read(func))
    Logger.semans.append(func)
    src = dedent(getsource(func))
    # Logger.rule_srcs.append(src[src.index('def'):])
    Logger.rule_srcs.append(src)
    return func


L = lex
R = rule

@G
class E:

    lex(IGNORED=r' ')
    lex(IGNORED=r'\t')

    lex(PLUS=r'\+', p=1)
    lex(TIMES=r'\*', p=2)
    lex(L=r'\(')
    lex(R=r'\)')

    @lex(NUM=r'[1-9]\d*', p=0)
    def _(val):
        return int(val)

    @rule
    def E(E, PLUS, T):
        return E + T
    @rule
    def E(T):
        return T

    @rule
    def T(T, TIMES, F):
        return T * F
    @rule
    def T(F):
        return F

    @rule
    def F(NUM):
        return NUM
    @rule
    def F(L, E, R):
        return E

    
from pprint import pprint


# # pprint(E.__dict__)
# pprint(E)

# # pprint(getsource(E['lex_srcs']['NUM']))
# s1 = (getsource(E['lex_hdls']['NUM']))
# s2 = (getsource(E['rule_list'][0]))


# pprint(s1)
# pprint(s2)

# md = ast.parse(dedent(s1))
# pprint(md.body[0].__dict__)
# fd = md.body[0]
# fd.decorator_list = []
# ctx = {}
# co = compile(ast.Module([fd]), '<ast>', 'exec')
# exec(co, globals(), ctx)
# pprint(ctx)


pprint(E)

p = LALR(E)

# print(p.parse('3 + 2 * (5 + 1)'))
# print(p.interpret('3 + 2 * (5 + 1)'))

(p.parse('3 + 2 * (5 + 1)'))
(p.interpret('3 + 2 * (5 + 1)'))

s1 = dedent(getsource(E.semans[3]))

pprint(s1)
import dis
# dis.dis(E.semans[3].__code__)

def _fake_lex(*a):
    def dec(func):
        return func
    return dec

def _fake_rule(func):
    return func

fake_ctx = {'lex': _fake_lex,
            'rule': _fake_rule }
ctx = fake_ctx
co = exec(s1, globals(), ctx)

pprint(ctx)
pprint(ctx['T'](4, None, 5))
# pprint(Logger.__dict__)
