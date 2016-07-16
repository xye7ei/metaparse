from metaparse import *

from collections import namedtuple

Node = namedtuple('Node', 'car cdr')

class Node(object):

    def __init__(self, car, cdr):
        self.car = car
        self.cdr = cdr

    def __repr__(self):
        return '({})'.format(self.car)

    def __iter__(self):
        yield self.car
        yield self.cdr

    def pop(self):
        if self is not NIL:
            value = self.car
            self.cdr = self.cdr.cdr
            return value
        else:
            raise ValueError('Pop from empty.')


NIL = Node(None, None)


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

    # def parse_many(self, inp, interp=False):

    #     push, pop = list.append, list.pop
    #     table = self.table
    #     G = self.grammar
    #     #
    #     agenda = [([], [(PREDICT, G.top_symbol)])]
    #     results = []
    #     #
    #     for k, tok in enumerate(G.tokenize(inp, with_end=True)):
    #         at, look, lexeme = tok
    #         agenda1 = []
    #         while agenda:
    #             (astk, pstk) = agenda.pop(0)
    #             if not pstk and len(astk) == 1: # and tok.symbol == END:
    #                 # Deliver partial result?
    #                 if tok.is_END():
    #                     results.append(astk[0])
    #             else:
    #                 act, actor = pstk.pop(0)
    #                 # Prediction
    #                 if act is PREDICT:
    #                     if actor in G.nonterminals:
    #                         if look in table[actor]:
    #                             for rule in table[actor][look]:
    #                                 nps = [(PREDICT, x) for x in rule.rhs] + [(REDUCE, rule)]
    #                                 agenda.append((astk[:], nps + pstk))
    #                         # NULLABLE
    #                         if EPSILON in table[actor]:
    #                             erule = table[actor][EPSILON][0]
    #                             arg = ParseTree(erule, [])
    #                             agenda.append((astk + [arg], pstk[:]))
    #                     else:
    #                         assert isinstance(actor, str)
    #                         if look == actor:
    #                             astk.append(tok)
    #                             agenda1.append((astk, pstk))
    #                         else:
    #                             # # May report dead state here for inspection
    #                             # print('Expecting \'{}\', but got {}: \n{}\n'.format(
    #                             #     actor,
    #                             #     tok,
    #                             #     pp.pformat((astk, pstk))))
    #                             # # BUT this can be quite many!!
    #                             pass
    #                 # Completion
    #                 else:
    #                     assert isinstance(actor, Rule)
    #                     subs = []
    #                     for _ in actor.rhs:
    #                         subs.insert(0, astk.pop())
    #                     astk.append(ParseTree(actor, subs))
    #                     agenda.append((astk, pstk))
    #         if agenda1:
    #             agenda = agenda1

    #     if interp:
    #         return [res.translate() for res in results]
    #     else:
    #         return results

    def parse_many(self, inp, interp=False):

        G = self.grammar
        T = self.table

        gss = Node(G.top_symbol, NIL)
        toplst = [gss]

        # for k, tok in enumerate(G.tokenize(inp, with_end=True)):
        toker = G.tokenize(inp, with_end=True)

        while toplst:
            at, look, lexeme = next(toker)
            z = 0
            while z < len(toplst):
                x, xs = toplst[z]
                if x in G.nonterminals: 
                    toplst.pop(z)
                    if look in T[x]:
                        for rule in T[x][look]:
                            xs1 = xs
                            for y in reversed(rule.rhs):
                                xs1 = Node(y, xs1)
                            toplst.append(xs1)
                else:
                    z += 1
            # Now toplst contains only terminals
            assert all(x in G.terminals for x in toplst)


class S(metaclass=GLL.meta):
    a, b = r'ab'
    def S(A, B): return
    def A(a): return
    def A(X): return
    def B(b): return
    def X(a): return

S.parse_many('ab')
# class S(metaclass=GLL.meta): 
#     a = r'a'
#     b = r'b' 
#     def S(A, B): pass
#     def A(a)   : pass
#     def A(S)   : pass
#     def B(b)   : pass


pp.pprint(S)
pp.pprint(S.table)
