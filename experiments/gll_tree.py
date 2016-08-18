from metaparse import *

@meta
class GLL(ParserBase):
    """This implemenation of GLL parser can handle left-recursion,
    left-sharing problem and simple loops properly. However, *hidden
    left recursion* must be handled with special caution and is
    currently not supported.

    """

    def __init__(self, grammar):
        super(GLL, self).__init__(grammar)
        assert hasattr(self, 'lexer')
        self.grammar = grammar

    def _find_hidden_left_rec(self):
        G = self.grammar
        raise NotImplementedError()

    def recognize(self, inp, interp=False):
        """Discover a recursive automaton on-the-fly.

        - Transplant a prediction tree whenever needed.

        - Tracing all states, especially forked states induced by
          expanded nodes.

        """

        G = self.grammar

        toks = []

        toplst = G.pred_tree(G.top_symbol, BOTTOM)
        stklst = [[] for _ in toplst]

        for k, tok in enumerate(G.tokenize(inp, with_end=True)):

            toks.append(tok)
            toplst1 = []

            # Transitive closure on :toplst:
            z = 0
            while z < len(toplst):
                n = (act, x), nxt = toplst[z]
                if act is PREDICT:
                    # Expand
                    if x in G.nonterminals:
                        # Transplant new prediction tree onto
                        # current nonterminal node.
                        for m in G.pred_tree(x, nxt):
                            toplst.append(m)
                    # Match
                    else:
                        if x == tok.symbol:
                            toplst1.append(nxt)
                elif act is REDUCE:
                    # Skipping nullable reduction.
                    assert isinstance(nxt, ExpdNode)
                    if nxt.value == G.top_symbol:
                        if tok.is_END():
                            print('Full recognition on: \n{}'.format(pp.pformat(toks[:k])))
                        else:
                            print('Partial recognition on: \n{}.'.format(pp.pformat(toks[:k])))
                    for nnxt in nxt.forks:
                        if nnxt is not BOTTOM:
                            # Check whether `nnxt` is already in the
                            # top list to avoid repeatedly appending
                            # existing nodes.
                            # But direct comparison between nodes may
                            # lead to nontermination.
                            if id(nnxt) not in (id(m) for m in toplst):
                                toplst.append(nnxt)
                else:
                    raise

                z += 1

            toplst = toplst1

    def parse_many(self, inp, interp=False):
        """Discover a recursive automaton on-the-fly.

        - Transplant a prediction tree whenever needed.

        - Tracing all states, especially forked states induced by
          expanded nodes.

        """
        global START, PREDICT, REDUCE, ACCEPT

        G = self.grammar
        L = self.lexer

        stk_btm = START
        # `bottom` comes after the reduction of top-symbol
        bottom = ExpdNode(ACCEPT, [])

        # GSS push
        push = Node

        # (<active-node>, <cumulative-stack>)
        toplst = [(n, stk_btm)
                  for n in G.pred_tree(G.top_symbol, bottom)]
        results = []

        toks = []

        for k, tok in enumerate(L.tokenize(inp, with_end=True)):
            toks.append(tok)

            toplst1 = []
            # Start from current node in top list and search the
            # active token.

            # Memoization to avoid cycles!
            # FIXME:
            # - explored should be prepared for each path! NOT shared!
            srchs = [(n, {}, stk) for n, stk in toplst]

            while srchs:
                n, expdd, stk = srchs.pop()
                if n is bottom:
                    # print(stk.to_list()[::-1])
                    if tok.is_END():
                        results.append(stk.to_list()[::-1])
                else:
                    (act, arg), nxt = n
                    if act is PREDICT:
                        X = arg
                        if X in G.terminals:
                            if X == tok.symbol:
                                # toplst1.append((nxt, stk + [tok.value]))
                                toplst1.append((nxt, push(tok.value, stk)))
                        else:
                            #
                            if X in expdd:
                                expdd[X].forks.append(nxt)
                            # Plant tree.
                            # FIXME: infinite planting?
                            else:
                                for m in G.pred_tree(X, nxt):
                                    # srchs.append((m, stk))
                                    # Prediction information not directly available.
                                    # FIXME: may store this in prediction trees.
                                    srchs.append((m, expdd, stk))
                    else:
                        rl = arg
                        assert isinstance(nxt, ExpdNode), nxt
                        m, fks = nxt
                        # FIXME:
                        if m not in expdd:
                        # if 1:
                            # expdd[m] = nxt
                            for fk in fks:
                                # srchs.append((fk, stk + [x, nxt.value]))
                                # Mind the push order!
                                # - ExpdNode redundant in stack, thus not pushed.
                                expdd1 = dict(expdd)
                                expdd1[m] = nxt
                                srchs.append((fk, expdd1, push(rl, stk)))

            if not toplst1:
                if not tok.is_END():
                    raise ParserError('Failed after reading \n{}.'.format(
                        pp.pformat(toks)
                    ))
            else:
                toplst = toplst1

        return results
