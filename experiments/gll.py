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

        bottom = ACCEPT
        toplst = G.pred_tree(G.top_symbol, bottom)
        stklst = [[] for _ in toplst]

        for k, tok in enumerate(G.tokenize(inp, with_end=True)):
        # for k, tok in enumerate(G.tokenize(inp, with_end=False)):

            toks.append(tok)
            toplst1 = []
            # stklst1 = []

            # Transitive closure on :toplst:
            z = 0
            while z < len(toplst):
                n = (act, x), nxt = toplst[z]
                # stk = stklst[z]
                if act is PREDICT:
                    if x in G.nonterminals:
                        # Transplant new prediction tree onto
                        # current nonterminal node.
                        sub_pred = G.pred_tree(x, nxt)
                        for m in sub_pred:
                            toplst.append(m)
                            # stklst.append(stk[:])
                    else:
                        if x == tok.symbol:
                            toplst1.append(nxt)
                            # stklst1.append(stk + [tok.value])
                elif act is REDUCE:
                    assert isinstance(nxt, ExpdNode)
                    if nxt.value == G.top_symbol:
                        if tok.is_END():
                            print('Full recognition on: \n{}'.format(pp.pformat(toks[:k])))
                            # return [stk for stk in stklst if stk[-1] == G.top_symbol]
                            # return True
                        else:
                            print('Partial recognition on: \n{}.'.format(pp.pformat(toks[:k])))
                            pass
                    # TODO
                    for nnxt in nxt.forks:
                        if nnxt is not bottom and nnxt not in toplst:
                            toplst.append(nnxt)
                            # stklst.append(stklst[z] + [':%s' % act.value])
                else:
                    raise

                z += 1

            toplst = toplst1
            # if tok.is_END():
            #     return stklst
            # else:
            #     stklst = stklst1

    def parse_many(self, inp, interp=False):
        """Discover a recursive automaton on-the-fly.

        - Transplant a prediction tree whenever needed.

        - Tracing all states, especially forked states induced by
          expanded nodes.

        """
        global START, PREDICT, REDUCE, ACCEPT

        G = self.grammar

        stk_btm = START
        bottom = ExpdNode((ACCEPT, G.top_symbol), [])

        # GSS push
        push = Node

        # (<active-node>, <cumulative-stack>)
        toplst = [(n, stk_btm)
                  for n in G.pred_tree(G.top_symbol, bottom)]
        toplsts = []

        for k, tok in enumerate(G.tokenize(inp, with_end=True)):

            toplst1 = []
            # Start from current node in top list and search the
            # active token.

            # Memoization to avoid cycles!
            # FIXME:
            # - explored should be set for each path! NOT shared!
            srchs = [(n, {}, stk) for n, stk in toplst]

            while srchs:
                n, expdd, stk = srchs.pop()
                if n is bottom:
                    print(stk.to_list()[::-1])
                    # yield stk
                else:
                    (act, x), nxt = n
                    if act is PREDICT:
                        if x in G.terminals:
                            if x == tok.symbol:
                                # toplst1.append((nxt, stk + [tok.value]))
                                toplst1.append((nxt, push(tok.value, stk)))
                        else:
                            #
                            if x in expdd:
                                expdd[x].forks.append(nxt)
                            # Plant tree.
                            # FIXME: infinite planting?
                            else:
                                for m in G.pred_tree(x, nxt):
                                    # srchs.append((m, stk))
                                    # Prediction information not directly available.
                                    # FIXME: may store this in prediction trees.
                                    srchs.append((m, expdd, stk))
                    else:
                        assert isinstance(nxt, ExpdNode), nxt
                        m, rst = nxt
                        if m not in expdd:
                            expdd[m] = nxt
                            for fk in nxt.forks:
                                # srchs.append((fk, stk + [x, nxt.value]))
                                # Mind the push order!
                                # - ExpdNode redundant in stack, thus not pushed.
                                srchs.append((fk, expdd, push(x, stk)))

            toplst = toplst1
            # toplsts.append(toplst1)

        # # Find traverse
        # return toplsts[-1]

