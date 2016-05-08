import grammar
import pprint as pp

from collections import namedtuple
from collections import OrderedDict

class LALR(grammar.Grammar):

    """
    Extend base class Grammar into a LALR parser. The CLOSURE algorithm differs
    from the default one defined in super class Grammar.
    """
    DUMMY = '\0'

    def __init__(self, lexes_rules):

        super(LALR, self).__init__(lexes_rules) 
        _top0 = self.rules[0]

        # Trail `END` token for LR(1) grammar.
        _top = grammar.Rule.raw(
            _top0.lhs,
            _top0.rhs[:1],
            lambda x: x
        )
        self.rules[0] = _top

        self.calc_lalr_item_sets()

    def __repr__(self):
        return 'LALR-{}'.format(super(LALR, self).__repr__())

    def close_default(G, I):
        """I :: [G.Item]           # without lookaheads

        This can be done before calculating LALR-Item-Sets, thus avoid
        computing closures repeatedly by applying the virtual dummy
        lookahead(`#` in the dragonbook). Since this lookahead must
        not be shared by any symbols within any instance of Grammar, a
        special value is used as the dummy(Not including None, since
        None is already used as epsilon in FIRST set). For similar
        implementations within lower-level language like C, this value
        can be replaced by any special number which would never
        represent a unicode character.

        """
        C = [(itm, LALR.DUMMY) for itm in I]
        z = 0
        while z < len(C):
            itm, a = C[z]
            if not itm.ended():
                for j, jrl in enumerate(G.rules):
                    if itm.active() == jrl.lhs:
                        jlk = []
                        beta = itm.over_rest()
                        for X in beta + [a]:
                            for b in G.first(X):
                                if b and b not in jlk:
                                    jlk.append(b)
                            if None not in G.first(X):
                                break
                        for b in jlk:
                            jtm = G.Item(j, 0)
                            if (jtm, b) not in C:
                                C.append((jtm, b))
            z += 1
        return C

    def calc_lalr_item_sets(G):
        Ks = [[G.Item(0, 0)]]   # Kernels
        goto = []
        spont = []
        propa = []
        Cs = []

        # Calculate Item Sets, GOTO and propagation graph in one pass.
        i = 0
        while i < len(Ks):

            K = Ks[i]
            C = G.close_default(K)
            Cs.append(C)

            # Use OrderedDict to preserve order of finding
            # goto's, which should be the same order with
            # the example in textbook.
            igoto = OrderedDict()
            # Should be defaultOrderedDict
            # ispont = defaultdict(set)
            ispont = OrderedDict()
            ipropa = []

            # For each possible goto relation, conclude its
            # corresponded spont/propa property (Dichotomy).
            for itm, a in C:
                # Prepare source of goto.
                if itm not in ispont:
                    ispont[itm] = set()
                # If item has goto (A -> α.Xβ)
                if not itm.ended():
                    X = itm.active()
                    jtm = itm.shifted()
                    # Avoid making move to complete augmented top
                    # rule.
                    if X not in igoto:
                        igoto[X] = []
                    if jtm not in igoto[X]:
                        igoto[X].append(jtm)
                    if a != LALR.DUMMY:
                        # spontaneous from `itm to its
                        # corresponding GOTO target.
                        ispont[itm].add(a)
                    else:
                        # propagation from `itm to its
                        # corresponding GOTO target.
                        ipropa.append(itm)

            # Register local goto into global goto.
            goto.append({})
            for X, J in igoto.items():
                # The Item-Sets should be treated as UNORDERED! So
                # sort J to identify the Lists with same items,
                # otherwise these Lists are differentiated due to
                # ordering, which though strengthens the power of LALR
                # grammar, but loses LALR`s characteristics.
                J = sorted(J, key=lambda i: (i.r, i.pos))
                if J not in Ks:
                    Ks.append(J)
                j = Ks.index(J)
                goto[i][X] = j

            spont.append(ispont)
            propa.append(ipropa)

            i += 1

        # The very top of spontaneous lookahead, which is NOT included
        # in the process above, since END symbol is not covered as an
        # legal lookahead all above.
        spont[0][G.Item(0, 0)] = {grammar.END}

        # All the kernel lookahead are almost determined by the
        # "directly spontaneously one-step propagation", but further
        # propagation should be done to garantee all the potential
        # valid lookaheads.
        b = 1
        while True:
            brk = True
            for i in range(len(Ks)):
                # Update Kernel-Set[i]
                Cas = G.close_default(Ks[i])
                for itm in Ks[i]:
                    for ctm, a in Cas:
                        if a == LALR.DUMMY:
                            lks = spont[i][itm]
                        else:
                            lks = [a]
                        for lk in lks:
                            # 
                            if lk not in spont[i][ctm]:
                                spont[i][ctm].add(lk)
                                brk = False
                            # if not ctm.ended():
                            #     X = ctm.active()
                            #     j = goto[i][X]
                            #     jtm = ctm.shifted()
                            #     if lk not in spont[j][jtm]:
                            #         spont[j][jtm].add(lk)
                            #         brk = False
                # Update propagation for each goto target.
                for itm in propa[i]:
                    if not itm.ended():
                        X = itm.active()
                        for lk in spont[i][itm]:
                            j = goto[i][X]
                            jtm = itm.shifted()
                            if lk not in spont[j][jtm]:
                                # print('In propagation at', i, ' adding ', a, ' to ', j, "'s", jtm)
                                spont[j][jtm].add(lk)
                                brk = False
            if brk:
                G.passes = b
                break
            else:
                b += 1

        G.Ks = Ks
        G.GOTO = goto
        G.propa = propa
        G.table = spont
        G.Cs = Cs

        # spont has included all the non-kernel items which are not
        # necessary if spont registers only the target, not the
        # source.  A table representation covering only kernel items
        # to preserve for simplicity.
        G.ktable = []
        for i, K in enumerate(Ks):
            klk = {}
            for k in K:
                klk[k] = spont[i][k]
            G.ktable.append(klk)

        # Construct ACTION table

        # SHIFT for non-ended to consume terminal for transition. 
        ACTION = [{} for _ in G.table]
        for i, xto in enumerate(G.GOTO):
            for a, j in xto.items():
                if a in G.terminals:
                    ACTION[i][a] = ('shift', j)

        # REDUCE for ended to reduce. 
        conflicts = []
        for i, cik in enumerate(G.table):
            for itm, lks in cik.items():
                if itm.ended():
                    for lk in lks:
                        if lk in ACTION[i]:
                            conflicts.append((i, lk, itm))
                        else:
                            ACTION[i][lk] = ('reduce', itm)
                if itm == G.Item(0, 1):
                    ACTION[i][grammar.END] = ('accept', None)

        if conflicts:
            msg = ''
            for i, lk, itm in conflicts:
                msg = '\n'.join([
                    '! LALR-Conflict raised:',
                    '  - in ACTION[{}]: '.format(i),
                    '{}'.format(pp.pformat(ACTION[i], indent=4)),
                    "  * conflicting action on token {}: ".format(repr(lk)),
                    "    {{{}: ('reduce', {})}}".format(repr(lk), itm)
                ])
            print(msg)

        G.conflicts = conflicts
        G.ACTION = ACTION

    def parse(G, inp, interp=False):

        """Perform table-driven deterministic parsing process. Only one parse
        tree is to be constructed.

        If `interp` mode is turned True, then a parse tree is reduced
        to semantic result once its sub-nodes are completed. Otherwise
        the return the parse tree as the result.

        """

        trees = []
        ss = [0]

        lexer = G.tokenize(inp)

        # Parsing process.
        tok = next(lexer)
        while 1:
            i = ss[-1]
            if tok.symb not in G.ACTION[i]:
                msg =  'LALR - Ignoring syntax error by {}'.format(tok)
                msg += '\n  Current state stack: {}'.format(ss)
                msg += '\n'
                print(msg)
                tok = next(lexer)
            else:
                act, arg = G.ACTION[i][tok.symb]

                # SHIFT
                if act == 'shift':
                    trees.append(tok.val)
                    ss.append(G.GOTO[i][tok.symb])
                    # Go on iteration/scanning
                    tok = next(lexer)

                # REDUCE
                elif act == 'reduce':
                    rtm = arg
                    ntar = rtm.target()
                    subts = []
                    for _ in range(rtm.size()):
                        subt = trees.pop()
                        subts.insert(0, subt)
                        ss.pop()
                    # Any need to compact singleton Tree into the element?
                    # if len(subts) == 0:
                    #     tree = ntar
                    # elif len(subts) == 1:
                    #     tree = (ntar, subts[0])
                    # else:
                    #     tree = (ntar, subts)
                    if interp:
                        tree = rtm.eval(*subts)
                    else:
                        tree = (ntar, subts)
                    trees.append(tree)
                    # New got symbol is used for shifting.
                    ss.append(G.GOTO[ss[-1]][ntar])

                # ACCEPT
                elif act == 'accept':
                    return trees[-1]

                else:
                    raise ValueError('Invalid action {} on {}'.format(act, arg))

        raise ValueError('No enough token for completing the parse. ')

    def interprete(G, inp):
        return G.parse(inp, interp=True)


class lalr(grammar.cfg):

    """
    This metaclass directly generates a LALR parser for the Grammar
    declaration in class manner. 
    """

    def __new__(mcls, n, bs, kw):
        lxs_rls = super(lalr, mcls).__new__(mcls, n, bs, kw)
        return LALR(lxs_rls)



if __name__ == '__main__':

    class Glrval(metaclass=grammar.cfg):

        EQ   = r'='
        STAR = r'\*'
        ID   = r'[A-Za-z_]\w*'
        DUM  = r'\d+'           # For checking unused token as warning. 

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

    GP = LALR(Glrval)

    import pprint as pp

    pp.pprint(list(enumerate(GP.ACTION)))

    # No errors/problem should be raised. 
    GP.parse('id')
    GP.parse('*id')
    GP.parse("id=id")
    GP.parse('id=*id')
    GP.parse('**x=y')

    # print('\n\nNow test constructed parser for errors in input.')
    # print('\nTest 1: waiting to report ingorance of `p`...\n')
    # p1 = GP.parse('**o p =*q')
    # pp.pprint(p1)
    # print('\nTest 2: waiting to report ingorance of `b` and `d`...\n')
    # p2 = GP.parse('**a b =* *c d')
    # pp.pprint(p2)

    pp.pprint(GP.propa)
