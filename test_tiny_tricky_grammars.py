import pprint as pp

from metaparse import *


class A(metaclass=Earley.meta):
    """Trick of this grammar:
        A -> A B
        A ->
        B ->
    where
        A => A B^n => A {self-LOOP!} => ε

    That is, before consuming the first token, the trees

      (A, []);
      (A, [(A, []),
           (B, [])]);
      (A, [(A, [(A, []),
                (B, [])])]);

      ... and infinitely many

    should all be completed with validity at the 0'th position. In
    other words, there is a CYCLE in the Graph-Structured-Stack
    representing the parse forest.

    This means the on-the-fly eager construction of parse trees may
    fail with non-termination.

    For practical purpose, such kind of grammar may be of nonsense and
    warning report should be generated.

    """
    b = r'b'
    def A(A, B): return
    def A(): return
    def B(b): return
    def B(): return

# pp.pprint(A.recognize('   b '))
# pp.pprint(A.parse_chart('   b '))
# # res = A.parse('  b  ')
# res = A.parse_forest('  b  ')
# pp.pprint(res)
# assert 0


class G(metaclass=cfg):
    """Ambigious grammar containing self-LOOP.

    G => G A B => G ε ε => G
    
    (G -> .) induces completion of (G -> G A B.)
    """ 
    a = r'a'
    def G(G, A, B): return
    def G(a)       : return
    def A(a, A)   : return
    def A()       : return
    def B(a)      : return
    def B()       : return


# p_G = Earley(G)
# p_G = GLR(G)
# p_G = GLL(G)
# print(*p_G.tokenize('  a a', with_end=True))
# pp.pprint(p_G.recognize('  a  a'))
# pp.pprint(p_G.parse('  a  a'))
# assert 0


class L(metaclass=cfg):
    """Ambigious grammar containing mutual-LOOP.

    (L -> M.) completes (M -> L.), again completes (L -> M.),
    thus non-termination."""
    a = r'a'
    def L(M): return
    def M(L): return
    def M(a): return

# p_L = Earley(L)
# pp.pprint(p_L.recognize('a'))
# p_L.parse('a')
# assert 0


class S(metaclass=cfg):
    """Ambiguous grammar with strong ambiguity, but no LOOPs."""
    u = r'u'
    def S(A, B, C) : pass
    def A(u)       : pass
    def A()        : pass
    def B(E)       : pass
    def B(F)       : pass
    def C(u)       : pass
    def C()        : pass
    def E(u)       : pass
    def E()        : pass
    def F(u)       : pass
    def F()        : pass


# p_S = Earley(S)
# p_S = GLR(S)
p_S = GLL(S)
# res = p_S.parse('u')
pp.pprint(p_S.parse_many('u'))
assert 0


# class aSa(metaclass=cfg):
#     """A grammar which can trick LL(1)'s backtracking."""
#     a = r'a'
#     def S(a_1, S, a_2):
#         return (a_1, S, a_2)
#     def S(a_1, a_2):
#         return (a_1, a_2)

# p_aSa = GLL(aSa)
# res = p_aSa.interpret('a      a')
# res = p_aSa.interpret('a    a  a  a')
# res = p_aSa.interpret('a    a  a  a a a')
# print(res)


class S(metaclass=Earley.meta):
    """A special ambigious grammar partitioning x sequence, which
    demonstrates Jay Earley's error figuring out the method of
    constructing parse forest correctly, noted by Tomita.

    """
    IGNORED = r'\s+'
    x = r'x'
    def S(S_1, S_2):
        return [S_1, S_2]
    def S(x):
        return '!'


class A(metaclass=Earley.meta):
    """Ambigious grammar with more-to-one completion. Shared completee
    item leads to forking of parent items' stacks.

    """
    IGNORED = r'\s+'
    a = r'a'
    def A(B, C): return

    def B(D):    return
    def B(E):    return

    def C(F):    return
    def C(G):    return
    def C(H):    return

    def D(a):    return
    def E(a):    return
    def F(a):    return
    def G(a):    return
    def H(a):    return


