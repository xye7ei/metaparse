import preamble

from metaparse import *

table = []
refs = 0

class G(metaclass=cfg):

    EQ   = r'='

    STAR = r'\*'
    def STAR(lex: r'\*') -> 3:
        global refs
        refs += 1
        # return lex

    # ID   = r'[_a-zA-Z]\w*'
    def ID(lex: r'[_a-zA-Z]\w*') -> 3:
        table.append(lex)
        return lex

    # def ID(ID = r'[_a-zA-Z]\w*') -> 3:
        # table.add(ID)
        # return ID

    def S(L, EQ, R):
        return ('assign', L, R)

    def S(R):
        return ('expr', R)

    def L(STAR, R):
        return ('deref', R)

    def L(ID):
        return ID

    def R(L):
        return L


if __name__ == '__main__':

    inp = '*a = **b'

    import pprint as pp

    p1 = LALR(G)
    p2 = GLR(G)

    s1 = p1.dumps()
    p1 = LALR.loads(s1, globals()) 
    s1 = p1.dumps()
    p1 = LALR.loads(s1, globals()) 
    s2 = p2.dumps()
    p2 = GLR.loads(s2, globals())

    r1 = p1.interpret_many(inp)[0]
    r2 = p2.interpret_many(inp)[0]

    assert r1 == r2 == \
        ('assign',
         ('deref', 'a'),
         ('deref', ('deref', 'b')))

    assert table == ['a', 'b', 'a', 'b'], table
    assert refs == 6

    s = p1.lexer.dumps()
    lexer = Lexer.loads(s, globals())
    pp.pprint([*lexer.tokenize(inp, True)])

    assert refs == 9
    pp.pprint(p1)
    pp.pprint(p1.__dict__)