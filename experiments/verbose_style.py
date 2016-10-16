import preamble
from metaparse import *


class LexLogger(object):

    def __init__(self):
        self.lexes = []
        self.lexpats = []
        self.lexhdls = {}
        self.prece = {}

    def __call__(self, p=None, **kw):
        name, pat = kw.popitem()
        assert not kw
        self.lexes.append(name)
        self.lexpats.append(pat)
        if p is not None:
            assert isinstance(p, int)
            self.prece[name] = p
        return lambda func: self.lexhdls.__setitem__(name, func)


class RuleLogger(list):

    def __init__(self):
        self.rules = []
        self.semans = []

    def __call__(self, seman):
        self.rules.append(Rule.read(seman))
        self.semans.append(seman)


def gramo(func):
    "Transform function definition directly to grammar object."
    ll, rl = LexLogger(), RuleLogger()
    func(ll, rl)
    # g = metaparse.Grammar()
    return list(ll), ll.hdl, list(rl)

def verbose(func):
    ll, rl = LexLogger(), RuleLogger()
    func(ll, rl)
    g = Grammar(**ll.__dict__, **rl.__dict__)
    return g


@verbose
def Calc(lex, rule):

    lex(IGNORED = r' ')
    lex(IGNORED = r'\t')

    @lex(NUM = r'[0-9]+')
    def __(val):
        return int(val)

    lex(EQ  = r'=')
    lex(ID  = r'[_a-zA-Z]\w*')

    lex(POW = r'\*\*', p=3)
    lex(MUL = r'\*'  , p=2)
    lex(ADD = r'\+'  , p=1)

    @rule
    def assign(ID, EQ, expr):
        table[ID] = expr
        return expr

    @rule
    def expr(NUM):
        return int(NUM)

    @rule
    def expr(expr_1, ADD, expr_2):
        return expr_1 + expr_2

    @rule
    def expr(expr, MUL, expr_1):
        return expr * expr_1

    @rule
    def expr(expr, POW, expr_1):
        return expr ** expr_1


from pprint import pprint
# pprint(Calc)

# r, sm = Calc[-1][-1]
# r
# sm

pprint(Calc)
pCalc = LALR(Calc)

table = {}
pCalc.interpret('x  = 3 + 7')

pprint(table)


import inspect
# inspect.getsource(sm)
import textwrap
import ast

sm = Calc.semans[-1]

src = inspect.getsource(sm)
src1 = textwrap.dedent(src)
md = ast.parse(src1)
fo = md.body[0]
fo.decorator_list = []
ctx = {}
co = compile(ast.Module([fo]), '<ast>', 'exec')
exec(co, {}, ctx)
sm1 = ctx.pop(sm.__name__)

import dis, io
with io.StringIO() as i1, io.StringIO() as i2:
    dis.dis(sm, file=i1)
    i1.seek(0)
    s1 = i1.read()
    dis.dis(sm, file=i2)
    i2.seek(0)
    s2 = i2.read()

print(s1)
print(s2)
assert s1 == s2

import itertools
# textwrap.indent
textwrap.dedent

# def src_def(func):
#     rlns = inspect.getsourcelines(func)
#     lns = []
#     for ln in rlns:
        
