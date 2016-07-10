import pprint as pp

from metaparse import *


class A(metaclass=Earley.meta):
    b = r'b'
    def A(A, B): return
    def A(): return
    def B(b): return
    def B(): return

# pp.pprint(A.recognize('   b b'))
# pp.pprint(A.parse_chart('   b b'))
res = A.parse('  b  ')
# pp.pprint(res)
assert 0



class G(metaclass=cfg):
    """Ambigious grammar with Null-rules."""
    a = r'a'
    def S(S, A, B): return
    # def S()       : return  # What if NULLABLE????
    def S(a)       : return
    def A(a, A)   : return
    def A()       : return
    def B(a)      : return
    def B()       : return

p_G = Earley(G)
# p_G = GLR(G)
# p_G = GLL(G)
# print(*p_G.tokenize('  a a', with_end=True))
# pp.pprint(p_G.recognize('  a  a'))
# res = p_G.recognize('  a  a')
# res = p_G.parse('  a  a')
# pp.pprint(res)

assert 0


class S(metaclass=cfg):
    """Ambiguous grammar with strong ambiguity. """
    u = r'u'
    def S(S, B, C) : return
    def S(u)       : return
    def S()        : return
    def B(E)       : return
    def B(F)       : return
    def C(u)       : return
    def C()        : return
    def E(u)       : return
    def E()        : return
    def F(u)       : return
    def F()        : return


p_S = Earley(S)
res = p_S.parse('u')
pp.pprint(res) 
assert 0


class GIfThenElse(metaclass=cfg):
    """Ambigious grammar with Dangling-Else structure."""

    IGNORED = r'\s'
    IF      = r'if'
    THEN    = r'then'
    ELSE    = r'else'
    EXPR    = r'\d+'
    SINGLE  = r'[_a-zA-Z]\w*'

    def stmt(SINGLE):
        return SINGLE
    def stmt(IF, EXPR, THEN, stmt):
        return ('it', stmt)
    def stmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return ('ite', stmt_1, stmt_2)


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


# Ambiguous parsing
assert len(S.interpret('x  x   x')) == 2
assert len(A.interpret('a    a  ')) == 6

ear_ife = Earley(GIfThenElse)
gll_ife = GLL(GIfThenElse)

i3e1 = 'if 1 then if 71 then if 23 then if 987 then aa else bb'

assert len(ear_ife.interpret(i3e1)) == 4
assert ear_ife.interpret(i3e1) == gll_ife.interpret(i3e1) == [('ite', ('it', ('it', ('it', 'aa'))), 'bb'),
                                                               ('it', ('ite', ('it', ('it', 'aa')), 'bb')),
                                                               ('it', ('it', ('ite', ('it', 'aa'), 'bb'))),
                                                               ('it', ('it', ('it', ('ite', 'aa', 'bb'))))]

pp.pprint(gll_ife.interpret('if 1 then if 2 then if 3 then x else yy else zzz'))
