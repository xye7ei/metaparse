import preamble

from metaparse import *

from collections import namedtuple
from pprint import pprint


@meta
class WGLL(ParserBase):
    """Table driven GLL without left-recursion support."""

    def __init__(self, grammar):
        super(WGLL, self).__init__(grammar)
        self.grammar = grammar
        self._calc_gll1_table()

    def _calc_gll1_table(self):
        G = self.grammar
        table = self.table = {}
        for r, rule in enumerate(G.rules):
            lhs, rhs = rule
            if lhs not in table:
                table[lhs] = defaultdict(list)
            if rhs:
                for a in G.first_of_seq(rhs, EPSILON):
                    if a is not EPSILON:
                        table[lhs][a].append(r)
            else:
                table[lhs][EPSILON].append(r)

    def parse_many(self, inp):

        G = self.grammar
        L = self.lexer

        table = self.table
        agenda = [(Node((PREDICT, G.top_symbol), None), None)]

        pop = lambda x: x
        push = Node

        for k, tok in enumerate(L.tokenize(inp, True)):

            agenda1 = []

            while agenda:
                sstk, tstk = agenda.pop()
                # Pop from GSS.
                (act, arg), sstk = pop(sstk)
                if act == PREDICT:
                    X = arg
                    if X in G.nonterminals:
                        if tok.symbol in table[X]:
                            for r in table[X][tok.symbol]:
                                sstk1 = sstk
                                sstk1 = push((REDUCE, r), sstk1)
                                for Y in reversed(G.rules[r].rhs):
                                    sstk1 = push((PREDICT, Y), sstk1)
                                agenda.append((sstk1, tstk))
                        if EPSILON in table[X]:
                            r, = table[X][EPSILON]
                            sstk = push((REDUCE, r), sstk)
                            agenda.append((sstk, tstk))
                    else:
                        if tok.symbol == X:
                            agenda1.append((sstk, push(tok, tstk)))
                else:
                    r = arg
                    rl = G.rules[r]
                    subs = deque()
                    for _ in rl.rhs:
                        sub, tstk = pop(tstk)
                        subs.appendleft(sub)
                    t = ParseTree(rl, subs, G.semans[r])
                    if rl.lhs is G.top_symbol:
                        if tok.is_END():
                            yield t
                    else:
                        agenda.append((sstk, push(t, tstk)))

            agenda = agenda1


@grammar
def IF():
    IF = 'if'
    THEN = 'then'
    ELSE = 'else'
    EXPR = '\d+'
    STMT = '\w+'
    def stmt(STMT): pass
    def stmt(IF, EXPR, THEN, stmt, ELSE, stmt_1): pass
    def stmt(IF, EXPR, THEN, stmt): pass


g = WGLL(IF)
glr = GLR(IF)

pprint([*g.parse_many('if 1 then if 2 then a else b')])

# timeit [*g.parse_many('if 1 then if 2 then a else b')]
# timeit [*glr.parse_many('if 1 then if 2 then a else b')]


@grammar
def sexp():
    SYM = r'[^\[\]\(\)\{\}\s\t\v\n]+'
    # L, R = '()'
    L = r'\('
    R = r'\)'
    def sexp(SYM): pass
    def sexp(L, slist, R): pass
    def slist(): pass
    def slist(sexp, slist): pass

s = WGLL(sexp)
slr = GLR(sexp)

pprint([*s.parse_many('(a (b))')])

# [*s.parse_many('(a (b c) ((((((d)))))) ((e) f (g h)))')]
# timeit [*s.parse_many('(a (b c) ((((((d)))))) ((e) f (g h)))')]
# timeit [*slr.parse_many('(a (b c) ((((((d)))))) ((e) f (g h)))')]
