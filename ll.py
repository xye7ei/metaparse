# The general LL-parser should be best supported by utilizing
# different Item object other than simple Item definition.
# It is considered to be expressionwise like regular expression,
# which allows primitive expression like `+`, `?` or `*`.

import grammar
import pprint as pp

from collections import OrderedDict

class LL(grammar.Grammar):

    def __init__(self, lexes_rules):
        """Input is a tuple of (<lexicals>, <rules>)"""
        super(LL, self).__init__(lexes_rules)
        self.cons_ll_table()

    def __repr__(self):
        return 'LL-{}'.format(super(LL, self).__repr__())

    def cons_ll_table(G):
        P = []
        for r in G.rules:
            pd = OrderedDict()
            for f in G.first(r.lhs):
                pd[f] = r
            P.append(pd)
        G.table = P

    def parse_with_table(G, inp: str) -> [tuple]:

        Item = namedtuple('Item', 'r subs')

        stk = [Item(0, [])]
        lexer = tokenize(inp)
        tk = next(lexer)
        while 1:
            itm = stk.pop()
            if itm.ended(): 
                tre = Tree(itm.tar, itm.subs)
                if not stk and tk == 'END':
                    return tre
                else:
                    stk[-1].subs.append(tre)
            else:
                stk.append(itm)
                X = itm.active()
                if X in G.nonterminals:
                    # Find row for rule. 
                    if tk in G.table(itm.r):
                        goto = G.table[itm.r][tk.symb]
                        stk.append(goto)
                    # If failed.
                    else:
                        print('Syntax error at {}'.format(tk))
                else:
                    if tk.symb == X:
                        itm.subs.append(tk.val)
                    else:
                        print('Syntax error at {}'.format(tk))
                tk = next(lexer)
        raise ValueError('Parse failed. ')

    def parse(G, inp):
        """
        Typical Recursive-Descendent-Parsing.
        TODO: Supporting EBNF for extensive usage and avoiding left-recursive.
        - Since most left-recursion are used for repeating objects, which can
          be covered by *-notation in regular expressions. 
        - Confer Parsing Expression Grammar.
        """

        FAIL = (None, -1) 

        tokens = list(G.tokenize(inp)) 
        i = 0

        memo = []
        # HOWTO: Use memoization to void repeated parsing
        # with the same rule at the same position! 
        def parse_rule(rl, i, dep=0):

            subs = []

            for X in rl.rhs:
                sub, i = parse_symb(X, i, dep) # Updating i.
                if sub != None:
                    subs.append(sub)
                else:
                    return FAIL

            return ((rl.lhs, subs), i)


        def parse_symb(X, i, dep=0):

            nonlocal inp

            if X in G.terminals: 
                # Match one token. 
                at, tok, tokval = tokens[i]
                return (tokval, i + 1) if tok == X else FAIL

            else:
                # Match alternative rules and get a parse forest.
                ress = []

                for rl in G.ntrie[X]:
                    tr, j = parse_rule(rl, i, dep+1)
                    if tr:
                        ress.append((tr, j))

                if not ress:
                    return FAIL
                else:
                    if len(ress) > 1:
                        print('Ambiguity - finding more trees. The first is taken.')
                    return ress[0]

        res, j = parse_symb(G.start_symbol, 0)

        if res:
            end_at = tokens[j][0] 
            if tokens[j][1] == 'END':
                msg = '\nFull parse for "{}".'.format(inp[:end_at])
            else:
                msg = '\nOnly partial parse "{}" from "{}".'.format(inp[:end_at], inp) 
            print(msg)
        else:
            print('Parse failed for "{}".'.format(inp))

        return res, j
        

if __name__ == '__main__':

    from grammar import cfg

    class S(metaclass=cfg):

        c = r'c'
        d = r'd'
        # e = r'e'

        def S(C_1, C_2):
            return 'S[  C[{}]  C[{}]  ]'.format(C_1, C_2)

        def C(c, C):
            return 'C[{} {}]'.format(c, C)

        def C(d):
            return 'C[d]'

    # sll = LL(*S)
    # # pp.pprint(sll.table)
    # sll.parse('dd')
    # # sll.parse('ccdcd')
    # inp = 'ccdcde'
    # list(sll.tokenize(inp))
    # sll.parse(inp)

    class Q(metaclass=cfg):
        a = r'a'
        def S(a_1, S, a_2): return
        def S(a_1, a_2): return

    pq = LL(Q)
    pp.pprint(pq.table)

    # Check deficiency of LL(1) parser w.R.t. deterministic choice
    # facing non-left-factored grammar with alternatives sharing
    # derivation symbols.

    # pq.parse('a')
    # pq.parse('aa')
    # pq.parse('aaa')
    # pq.parse('aaaa')
    # pq.parse('aaaaa')
    # pq.parse('aaaaaa')
    # pq.parse('aaaaaaaa')
    # pq.parse('aaaaaaaaaa')
    # pq.parse('aaaaaaaaaaaa')
    # pq.parse('aaaaaaaaaaaaaa')
    # pq.parse('aaaaaaaaaaaaaaaa')

    class SE(metaclass=cfg):

        NUM = r'\d+'
        OP  = r'\+'

        def E(E, OP, T): return
        def E(T): return

        def T(NUM): return

    se = LL(SE)
    # se.parse('3')
    # se.parse('3 + 5 + 7')
