from gll_tree import *
from pprint import pprint

@grammar
def E():
    ID = r'[_a-zA-Z]\w*'
    plus = r'\+'
    times = r'\*'
    left = r'\('
    right = r'\)'
    def E(E, plus, T)     : pass
    def E(T)              : pass
    def T(T, times, F)    : pass
    def T(F)              : pass
    def F(ID)             : pass
    def F(left, E, right) : pass

print('====================')
# pp.pprint(E.FIRST)
# pp.pprint(E.pred_tree('E'))
# pp.pprint([n.value for n in E.pred_tree('E')])
gll_E = GLL(E) 
# res = gll_E.recognize('a + b * c') 
gll_E.parse_many('a + b * c') 
# pp.pprint(res)
# assert 0
print('====================\n')


@grammar
def I():
    IF = r'if'
    THEN = r'then'
    ELSE = r'else'
    EXPR = r'\d+'
    SENT = r'\w+'
    def stmt(SENT): pass
    def stmt(IF, EXPR, THEN, stmt): pass
    def stmt(IF, EXPR, THEN, stmt, ELSE, stmt_1): pass

print('====================')
gll_I = GLL(I)
glr_I = GLR(I)
# tr = I.pred_tree('stmt')
# pprint(tr)
# raise
# res = gll_I.recognize('if 1 then if 2 then a else b')
# print(res)
# pprint(I.pred_tree('stmt'))
# gll_I.parse_many('if 1 then if 2 then a else b')
# glr_I.parse_many('if 1 then if 2 then a else b')
# pprint(gll_I.parse_many('if 1 then if 2 then a else b'))
# pprint(glr_I.parse_many('if 1 then if 2 then a else b'))
# res = [*gll_I.parse_many('if 1 then if 2 then a else b')]
# print(len(res))
# pp.pprint(res)
# pp.pprint(res)
print('====================\n')


print('====================')
@grammar
def L():
    "Grammar with left-recursion and no loops, "
    a = r'a'
    def S(A): pass
    def S(): pass
    def A(B): pass
    def B(S, a): pass

gll_L = GLL(L)
# gll_L.recognize('a a  a')
gll_L.parse_many('a a  a')
print('====================\n')
