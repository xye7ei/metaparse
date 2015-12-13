import grammar
import copy
import pprint as pp

from collections import namedtuple
from collections import OrderedDict

class GLR(grammar.Grammar):

    """GLR parser as an instance of non-deterministic parsers. Typically,
    non-deterministic parsers produce parse forest other than exactly
    one parse tree.

    This leads to the deterministic semantic application along the
    parse process UNABLE be performed with potential side-effects. To
    support such application, the parse tree in the forest should be
    modeled with corresponding semantic promises, such that one of the
    constructed parse trees can be chosen for application whilst the
    other are discarded.

    """

    def __init__(self, lxs_rls):
        super(GLR, self).__init__(lxs_rls)
        self._calc_glr_item_sets()

    def __repr__(self):
        return 'General-LR-{}'.format(super(GLR, self).__repr__())

    def _calc_glr_item_sets(G):

        """Calculate general LR-Item-Sets with no respect to look-aheads.
        Each conflict is registered into the parsing table. For
        practical purpose, these conflicts should be reported for the
        grammar writer to convey the conflicts and maybe experiment
        with potential ambiguity, thus achieving better inspection
        into the characteristics of the grammar itself.

        For LR(0) grammars, the performance of GLR is no worse than
        the LALR(1) parser.

        """

        G.Ks = Ks = [[G.Item(0, 0)]]
        G.GOTO = goto = []
        G.ACTION = acts = []

        # Fixed point computation. 
        z = 0
        while z < len(Ks):

            K = Ks[z]
            C = G.closure(K)
            iacts = {'reduce': [], 'shift': {}}
            igotoset = OrderedDict()

            for itm in C:
                if itm.ended():
                    iacts['reduce'].append(itm)
                else:
                    X = itm.active()
                    jtm = itm.shifted()
                    if X not in igotoset:
                        igotoset[X] = []
                    if jtm not in igotoset[X]:
                        igotoset[X].append(jtm)

            igoto = OrderedDict()
            for X, J in igotoset.items():
                if J not in Ks:
                    Ks.append(J)
                j = Ks.index(J)
                iacts['shift'][X] = j
                igoto[X] = j
            acts.append(iacts)
            goto.append(igoto)
            z += 1

    def parse(G, inp): 

        """
        When forking the stack, there may be some issues:
        - SHIFT consumes a token, while REDUCE consumes no.
        - As a result, the single-threaded generator can not satisfy
          the needed of feeding different stacks with token at different
          step;
        - So there are some possibilities to handle this:
        - The Overall-Stack-Set can be maintained as a queue. For each stack
        element in the Overall-Stack-Set, keep a position index in the input
        token sequence(or a teed generator) associated with each stack element.
        This allows virtual backtracking to another fork time-point when the
        active stack failed. 
            * As a result, the probes for each stack in Overall-Stack-Set can be
            done in a DFS/BFS/Best-FS manner. 
            * In case no lookahead information is incorperated, the GLR parser
            can keep track of all viable Partial Parsing all along the process. 
        """

        results = []
        lexer = G.tokenize(inp)
        tokens = list(lexer)
        prss = [[[0], [], 0]]          # prss :: [[State-Number, [Tree], InputPointer]]

        while prss:

            stk, trns, i = prss.pop() # stack top

            if i == len(tokens):
                # Now trns should be [Tree, END], where Tree is the
                # top of non-augmented grammar! So trns[0] means
                # exactly the non-augmented root tree.
                if len(trns) == 2 and trns[-1] == grammar.END_PAT:
                    results.append(trns[0])
                continue

            tok = tokens[i]
            stt = stk[-1]
            reds = G.ACTION[stt]['reduce']
            shif = G.ACTION[stt]['shift']

            # REDUCE
            # There may be multiple reduction options. Each option leads
            # to one fork of the parsing state.
            for ritm in reds:
                # Forking, copying State-Stack and Trns
                # Index of input remains unchanged. 
                frk = stk[:]
                # trs = copy.deepcopy(trns)
                trs = trns[:]
                subts = []
                for _ in range(ritm.size()):
                    frk.pop()
                    subts.insert(0, trs.pop())
                tar = ritm.target()
                ntr = (tar, subts)
                trs.append(ntr)

                frk.append(G.GOTO[frk[-1]][tar])
                prss.append([frk, trs, i]) # index i stays

            # SHIFT
            # There can be only 1 option for shifting given a symbol due
            # to the nature of LR automaton.
            if tok.symb in shif:
                stk.append(G.GOTO[stt][tok.symb])
                trns.append(tok.val)
                prss.append([stk, trns, i+1]) # index i increases

            # # ACCEPT
            # if len(stk) == 1 and tok.symb == grammar.END:
            #     results.append(trns)
            #     continue

            # ERROR
            if not reds and tok.symb not in shif and tok.symb != grammar.END:
                # Need any hints for tracing dead states? 
                # msg = '\nWithin parsing fork {}'.format(stk)
                # msg += '\nSyntax error ignored: {}.'.format(tok)
                # msg += '\nChoking item set : \n{}'.format(
                #     pp.pformat(G.closure(G.Ks[stk[-1]])))
                # msg += '\nExpected shifters: \n{}'.format(
                #     pp.pformat(shif))
                # print(msg)
                continue

        # if not results:
        #     print('No parse tree generated. Check ignored position. ')
        # elif len(results) > 1:
        #     print('Ambiguity raised: {} parse trns produced.'.format(len(results)))
        return results


class glr(grammar.cfg):

    def __new__(mcls, n, bs, kw):
        lxs_rls = super(glr, mcls).__new__(mcls, n, bs, kw)
        return GLR(lxs_rls)
    

if __name__ == '__main__':

    class Glrval(metaclass=grammar.cfg):

        EQ   = r'='
        STAR = r'\*'
        ID   = r'[A-Za-z_]\w*'

        def S(L, EQ, R):
            return ('assign', L, R)

        def S(R):
            return ('expr', R)

        def L(STAR, R):
            return ('deref', R)

        def L(ID):
            return 'id'

        def R(L):
            return L


    import pprint as pp
    print()
    # pp.pprint(Glrval.Ks)
    # pp.pprint(Glrval.ACTION)

    # pp.pprint(list(enumerate(Glrval.ACTION)))

    G = GLR(Glrval)
    # G.parse('id')
    # G.parse('*id')
    # G.parse("id=id")
    # G.parse('id=*id')
    # G.parse('**x=y')

    res = G.parse('*  *o  =*q')
    pp.pprint(G.ACTION)
    print('Result:')
    pp.pprint(res)
    # p1 = G.parse('**o p =*q')
    # pp.pprint(p1)
    # p2 = G.parse('**a b =* *c d')
    # pp.pprint(p2)

