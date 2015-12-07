import pprint as pp

import grammar
from collections import namedtuple
from collections import OrderedDict

Item = namedtuple('Item', 'rule i stk')
Item.__repr__ = lambda s: '[ {:<10} ->   {:<30} ]'.format(s.rule.lhs,
                                                     ' '.join(s.rule.rhs[:s.i]) \
                                                     + '.' + \
                                                     ' '.join(s.rule.rhs[s.i:]))
Item.make_shifted_with = lambda s, x: Item(s.rule, s.i + 1, s.stk + (x,))
Item.waiting = lambda s: None if s.i == len(s.rule.rhs) else s.rule.rhs[s.i]
Item.target = lambda s: s.rule.lhs
Item.value = lambda s: s.rule.seman(*s.stk)

def item2tree(it):
    tstk = [(it.target(), [], list(it.stk))]
    while tstk:
        # Peek stack top.
        node = tstk[-1]
        # If peeked tree already completed.
        if not node[2]:
            subh, subtl, _ = tstk.pop()
            # if len(subtl) == 1:
            #     subtl = subtl[0]
            if tstk:
                tstk[-1][1].append((subh, subtl))
            else:
                return (subh, subtl)
        # If peeked tree not completed.
        else:
            # If next argument to handle is an Item.
            # Then expand it as a tree to complete and push it onto stack.
            if isinstance(node[2][0], Item):
                jt = node[2].pop(0)
                tstk.append((jt.target(), [], list(jt.stk)))
            # If next argument to handle is not an Item, i.e. it's a leaf.
            # Then directly push it onto the preceding stack tree's
            # completed argument list.
            else:
                node[1].append(node[2].pop(0))

def eval_item(it):
    """ Applying post-order scheme to avoid recursion. """
    # item-stack :: [(Item<to be evaled>, []<calculated arg>, []<uncalculated arg>)]
    istk = [(it, [], list(it.stk))]
    while istk:
        f = istk[-1]
        # If the peeked Item has enough arguments calculated.
        if not f[2]:
            istk.pop()
            fval = f[0].rule.seman(*f[1])
            if istk:
                istk[-1][1].append(fval)
            else:
                return fval
        # If the peeked Item's next argument is an Item, which must be evaluated into a value first,
        # Then push it upon the stack.
        elif isinstance(f[2][0], Item):
            jt = f[2].pop(0)
            istk.append((jt, [], list(jt.stk)))
        # If hte peeked Item's next argument is not an Item, which is a value and can be shifted into
        # evaluated argument list.
        else:
            f[1].append(f[2].pop(0))

Item.to_tree = item2tree
Item.eval    = eval_item


class Earley(grammar.Grammar):

    def __init__(self, lexes_rules):

        super(Earley, self).__init__(lexes_rules)
        self.start_item = Item(self.start_rule, i=0, stk=())

    def __repr__(self):
        return '{}{}'.format('Earley', super(Earley, self).__repr__())

    def parse_process(G, inp : str) -> [[Item]]:

        """
        Parse a iterable of strings w.R.t. given grammar into states.
        - How to use the chart lazily??
        - How is the lazy overhead compared to strict?
          - Time?
          - Space?
        """

        S = [[(G.start_item, 0)]]

        for k, (at, lex, lexval) in enumerate(G.tokenize(inp)):

            # Note len(S[k]) is growing!
            # Find candidate items until no more added. 
            z = 0
            while z < len(S[k]):

                jt, j = S[k][z]

                # prediction for `jt , i.e. implicitly applying CLOSURE
                if jt.waiting() and jt.waiting() in G.nonterminals:
                    for prd in G.ntrie[jt.waiting()]:
                        new = (Item(prd, 0, ()), k)
                        if new not in S[k]:
                            S[k].append(new)

                # completion with `jt
                if not jt.waiting():
                    # HOWTO handle side-effects using semantics?
                    for it, i in S[j]:
                        if it.waiting():
                            if it.waiting() == jt.target():
                                new = (it.make_shifted_with(jt), i)
                                if new not in S[k]:
                                    S[k].append(new)

                z += 1

            # ONE-PASS
            # scanning for accumulated items through prediction and completion.
            # S grows on the fly. Preparing new empty state list for scanning.
            S.append([])
            for jt, j in S[k]:
                # Mind the two following cases are not mutually exclusive,
                # i.e. there may be successfully ended top item within intermediate
                # states. 
                # Top rule completed.
                if not jt.waiting() and jt.rule == G.start_rule and lex == grammar.END:
                    S[k+1].append(jt)
                # Normal rules.
                if jt.waiting() and jt.waiting() == lex:
                    # Here terminal consumers can be collected for reporting errors.
                    S[k+1].append((jt.make_shifted_with(lexval), j))

            if not S[k+1]:
                msg = '\nSyntax error at {}th position in input handling ({} : {}). '.format(at, lex, lexval)
                msg += '\nChoked states: {}'.format(pp.pformat(S[k]))
                # raise ValueError(msg)
                # Or ignoring error?
                # Ignoring the current lex symbol can be done by copying S[k]
                # to S[k+1], i.e. maintaining active states before meeting this lex.
                msg += '\nToken ignored and go on...'
                print(msg)
                S[k+1] = S[k]
            elif len(S[k+1]) > 1:
                # Here local ambiguity raises
                pass

        return S

    def parse(self, inp):
        "Return the final state parsing `inp`, based on `parse_result`."
        s = self.parse_process(inp)
        final = s[-1]
        if final:
            if len(final) > 1:
                print('Ambiguity raised.')
                print('{} parse trees produced.'.format(len(final)))
                print('Returning parse forest.')
        else:
            print('Parse failed.')
        return [fitm.to_tree() for fitm in final]

    def eval(self, inp):
        s = self.parse_result(inp)
        final = s[-1]
        if final:
            if len(final) == 1:
                return final[0].eval()
            else:
                print('Ambiguity raised.')
                print('Returning parse forest.')
                return [fitm[0].eval() for fitm in final]
        else:
            print('Parse faield')
            return s

if __name__ == '__main__':

    class S(metaclass=grammar.cfg):

        c = r'c'
        d = r'd'

        def S(C1, C2):
            return 'S[  C[{}]  C[{}]  ]'.format(C1, C2)

        def C(c, C):
            return 'C[{} {}]'.format(c, C)

        def C(d):
            return 'C[d]'

    p1 = Earley(S)

    class GArith(metaclass=grammar.cfg):

        PLUS  = r'\+'
        TIMES = r'\*'
        NUMB  = r'\d+'
        LEFT  = r'\('
        RIGHT = r'\)'

        def expr(expr, PLUS, term):
            'E ::= E + T'
            return expr + term

        def expr(term): 
            'E ::= T'
            return term

        def term(term, TIMES, factor):
            'T ::= T * F'
            return term * factor

        def term(factor):
            'T ::= F'
            return factor

        def factor(atom):
            'F ::= atom'
            return atom

        def factor(LEFT, expr, RIGHT):
            'F ::= ( E )'
            return expr

        def atom(NUMB):
            return int(NUMB)

    p2 = Earley(GArith)

    pp.pprint(p2.parse('3 + 2 * 5'))
    # GArith.eval('3 + 2 * 5 + 10 * 10')
    # pp.pprint(GArith.parse('3 + 2 * (5 + 10) * 10'))
    # print(GArith.eval('3 + 2 * (5 + 10) * 10'))
