from gll_tree import *

from pprint import pprint


class Foo:

    class S3(metaclass=cfg):
        """Grammar with loop, but without nullable. This does not hurt the
        naive algorithm which defends a cycle in top list.

        """
        a = 'a'
        def S(A): return
        def A(S): return
        def A(S, a): return
        def A(a): return

    @grammar
    def S4():
        """Hidden left-recursion. Note the incompletion when preparing
        back-pointer from [A].

        """
        a = 'a'
        b = 'b'
        c = 'c'
        def S(A, c): return
        def A(H, A, b): return
        def A(a): return
        def H(c): return
        def H(): return

    @grammar
    def S5():
        """Grammar with ever-growing nullable loop. It leads unending
        transplanting of prediction trees.

        """
        a = 'a'
        def S(S, T): return
        def S(): return
        def S(a): return
        def T(T, S): return
        def T(): return

    @grammar
    def S6():
        """Grammar with loops and nullables. It is hard to stop the predition
        tree from growing.

        """
        a = 'a'
        def S(S, T, U): return
        def T(T, U, S): return
        def U(U, S, T): return
        def S(): return
        def T(): return
        def U(): return
        def S(a): return


S3 = Foo.S3
S4 = Foo.S4
S5 = Foo.S5
S6 = Foo.S6

p3 = GLL(Foo.S3)
p4 = GLL(Foo.S4)
p5 = GLL(Foo.S5)
p6 = GLL(Foo.S6)

# g4 = GLR(Foo.S4)
e4 = Earley(Foo.S4)
pprint(p4.parse_many('cabc'))
# pprint(g4.parse_many('cabc'))
pprint(e4.parse_many('cabc'))
# l4 = LALR(Foo.S4)

# tr = S4.pred_tree('S')
# p4.recognize('cabc')

# p3.recognize('a  a  a')
# pprint(tr)

