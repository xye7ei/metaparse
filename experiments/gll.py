import preamble

from metaparse import *

from collections import namedtuple
from pprint import pprint


@meta
class WGLL(ParserBase):
    """Table driven GLL without left-recursion support."""

    def __init__(self, grammar):
        super(WGLL, self).__init__(grammar)
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
                        table[lhs][a].append(rule)
            else:
                table[lhs][EPSILON].append(rule)

    def parse_many(self, inp):
        G = self.grammar
        table = self.table
        agenda = [(Node((PREDICT, G.top_symbol), None), None)]

        pop = lambda x: x
        push = Node

        for k, tok in enumerate(G.tokenize(inp, True)):

            agenda1 = []

            while agenda:
                sstk, tstk = agenda.pop()
                # Pop from GSS.
                (act, arg), sstk = pop(sstk)
                if act is PREDICT:
                    X = arg
                    if X in G.nonterminals:
                        if tok.symbol in table[X]:
                            for rl in table[X][tok.symbol]:
                                sstk1 = sstk
                                sstk1 = push((REDUCE, rl), sstk1)
                                for Y in reversed(rl.rhs):
                                    sstk1 = push((PREDICT, Y), sstk1)
                                agenda.append((sstk1, tstk))
                        if EPSILON in table[X]:
                            rl = table[X][EPSILON][0]
                            sstk = push((REDUCE, rl), sstk)
                            agenda.append((sstk, tstk))
                    else:
                        if tok.symbol == X:
                            agenda1.append((sstk, push(tok, tstk)))
                else:
                    rl = arg
                    subs = deque()
                    for _ in rl.rhs:
                        sub, tstk = pop(tstk)
                        subs.appendleft(sub)
                    t = ParseTree(rl, subs)
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

# pprint([*g.parse_many('if 1 then a')])
# pprint([*g.parse_many('if 1 then a else b')])
pprint([*g.parse_many('if 1 then if 2 then a else b')])


@WGLL
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

pprint([*sexp.parse_many('(a (b))')])
