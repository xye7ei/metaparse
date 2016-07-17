# import preamble

from metaparse import *

from collections import namedtuple


@meta
class GLL(ParserBase):

    def __init__(self, grammar):
        super(GLL, self).__init__(grammar)
        self._calc_gll1_table()

    def _calc_gll1_table(self):
        G = self.grammar
        table = self.table = {}
        for r, rule in enumerate(G.rules):
            lhs, rhs = rule
            if lhs not in table:
                table[lhs] = defaultdict(list)
            if rhs:
                for a in G.first_star(rhs, EPSILON):
                    if a is not EPSILON:
                        table[lhs][a].append(rule)
            else:
                table[lhs][EPSILON].append(rule)

    def parse_many(self, inp, interp=False):

        raise NotImplementedError()


class S(metaclass=cfg):
    a, b = r'ab'
    def S(A, B): return
    def A(a): return
    def A(X): return
    def B(b): return
    def X(a): return


# gS = S.prediction_graph('S')
# gA = S.prediction_graph('A')
# gB = S.prediction_graph('B')

# print(gS)
# print(gA)
# print(gB)

@grammar
def S():
    a = r'a'
    b = r'b'
    c = r'c'
    def S(A, B): return
    def A(a): return
    def A(c): return
    def A(): return
    def A(S): return
    def B(b): return

print(S.PRED_TREE['S'])
print(S.PRED_TREE['A'])
print(S.PRED_TREE['B'])

print(S.FIRST['S'])
print(S.FIRST['A'])
print(S.FIRST['B'])
