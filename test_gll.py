from gll_raw import *


class S1(metaclass=cfg):
    a, b = r'ab'
    def S(A, B): return
    def A(a): return
    def A(X): return
    def B(b): return
    def X(a): return


class S2(metaclass=cfg):
    a, b = r'ab'
    def S(A, B): return
    def A(): return
    def A(A, a): return
    def B(b): return
    def B(): return



# class S4(metaclass=cfg):
class Foo:

    class S3(metaclass=cfg):
        a = 'a'
        def S(A): return
        def A(S): return
        def A(S, a): return
        def A(a): return

    @grammar
    def S4():
        a = 'a'
        def S(S, T, U): return
        def T(T, U, S): return
        def U(U, S, T): return
        def S(): return
        def T(): return
        def U(): return
        def S(a): return


# print('NULLABLE: ', S2.NULLABLE)

# print(S2.PRED_TREE['S'])
# print(S2.PRED_TREE['A'])
# print(S2.PRED_TREE['B'])


# # Normal
# p1 = GLL(S1)
# r = p1.recognize('ab')
# print(r)

# Left recursion
# print(S2.FIRST)
# pp.pprint(S2.NULLABLE)
# assert 0
# p2 = GLL(S2)
# p2.recognize('aaab')
# p2.recognize('aaabb')

# # Non-pure Loop is totally not problem for recognition
p3 = GLL(Foo.S3)
e3 = Earley(Foo.S3)
# Seems GLL spends only 1/3 time of Earley recognizing such input.
# print(S3.PRED_TREE['S'])
p3.recognize('aaaaaaaa')
e3.recognize('aaaaaaaa')
# LOOPed Grammar. Parse not terminated.
# rs = e3.parse_many('aaaaaaaa')
# print(rs)
assert 0

# # Pure loop is fatal for recognition
# # - 
# # -
# pp.pprint([*S4.find_loop('S^')])
# print(S4.pred_tree('S').__len__())
# pp.pprint(S4.pred_tree('S'))
# pp.pprint(S4.NULLABLE)
S4 = Foo.S4
pp.pprint(S4.DERIV1)
pp.pprint([*S4.find_loop('S')])

p4 = GLL(S4)
e4 = Earley(S4)
# print(S4.PRED_TREE['S'])
# Dead-Loop by GLL
# pp.pprint(p4.recognize('a'))
# No problem for Earley
# pp.pprint(e4.recognize('a'))


# class G_ite(metaclass=cfg):
#     IF = r'if'
#     THEN = r'then'
#     ELSE = r'else'
#     EXPR = r'\d+'
#     SINGLE = r'\w+'
#     def stmt(SINGLE): return
#     def stmt(IF, EXPR, THEN, stmt): return
#     def stmt(IF, EXPR, THEN, stmt, ELSE, stmt_1): return

# p3 = GLL(G_ite)
# p3.recognize('if 1 then if 2 then a else b')
