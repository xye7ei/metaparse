import preamble
import metaparse

class LexLog(list):

    def __init__(self):
        self.hdl = {}

    def hdl_log(self, name, func):
        self.hdl[name] = func

    def __call__(self, p=None, **kw):
        name, pat = kw.popitem()
        self.append((name, pat, p))
        return lambda func: self.hdl_log(name, func)

class RuleLog(list):
    def __call__(self, seman):
        r = metaparse.Rule.read(seman)
        self.append((r, seman))


def gramo(func):
    "Transform function definition directly to grammar object."
    ll, rl = LexLog(), RuleLog()
    func(ll, rl)
    # g = metaparse.Grammar()
    return list(ll), ll.hdl, list(rl)

@gramo
def Calc(L, R):

    L(IGNORED = r' ')
    L(IGNORED = r'\t')

    @L(NUM = r'[0-9]+')
    def NUM(val):
        return int(val)

    L(EQ  = r'=')
    L(ID  = r'[_a-zA-Z]\w*')

    L(POW = r'\*\*', p=3)
    L(MUL = r'\*'  , p=2)
    L(ADD = r'\+'  , p=1)

    @R
    def assign(ID, EQ, expr):
        table[ID] = expr
        return expr

    @R
    def expr(NUM):
        return int(NUM)

    @R
    def expr(expr_1, ADD, expr_2):
        return expr_1 + expr_2

    @R
    def expr(expr, MUL, expr_1):
        return expr * expr_1

    @R
    def expr(expr, POW, expr_1):
        return expr ** expr_1


from pprint import pprint
pprint(Calc)

r, sm = Calc[-1][-1]
r
sm

import inspect
inspect.getsource(sm)

import ast
md = ast.parse(inspect.getsource(sm).replace('\n    ', '\n'))
