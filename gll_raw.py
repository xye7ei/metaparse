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
                for a in G.first_of_seq(rhs, EPSILON):
                    if a is not EPSILON:
                        table[lhs][a].append(rule)
            else:
                table[lhs][EPSILON].append(rule)

    def recognize(self, inp, interp=False):
        """Discover a recursive automaton on-the-fly.

        - Transplant a prediction tree whenever needed.

        - Tracing all states, especially forked states induced by
          expanded nodes.

        """

        G = self.grammar

        toks = []
        toplst = G.pred_tree(G.top_symbol)

        acts = toplst[:]

        # Caution on nullable-cycle
        for k, tok in enumerate(G.tokenize(inp, with_end=True)):
            toks.append(tok)
            acts1 = []
            # Transitive closure on :acts:
            z = 0
            while z < len(acts):
                act = acts[z]
                if isinstance(act, Node):
                    if act.value in G.terminals:
                        if act.value == tok.symbol:
                            acts1.append(act.next)
                        else:
                            pass
                    else:
                        # Transplant new prediction tree onto
                        # current rest.
                        sub_pred = G.pred_tree(act.value, act.next)
                        acts.extend(sub_pred)
                elif isinstance(act, ExpdNode):
                    if act.value == G.top_symbol:
                        if tok.is_END():
                            # print('Full parse on: \n{}'.format(pp.pformat(toks[:k])))
                            return True
                        else:
                            # print('Partial parse on: \n{}.'.format(pp.pformat(toks[:k])))
                            pass
                    # TODO
                    # - Find cluster of neighbored ExpdNodes!
                    #   - which means EPSILON-transition in automata
                    # - Select one representative!
                    for nxt in act.forks:
                        if nxt not in acts:
                            acts.append(nxt)
                else:
                    pass

                z += 1

            acts = acts1
                        
    def parse_many(self, inp, interp=False):
        pass

