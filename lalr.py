import grammar
import pprint as pp

from collections import namedtuple
from collections import OrderedDict

class LALR(grammar.Grammar):

    """
    Abstraction of Context-Free Grammar.
    Open questions:
      - Need to register Nullable rules?
    """

    def __init__(self, lexes_rules):

        super(LALR, self).__init__(lexes_rules) 
        self.calc_lalr_item_sets()

    def __repr__(self):
        return 'LALR-{}'.format(super(LALR, self).__repr__())

    def close_default(G, I):
        """
        I :: [G.Item]           # without lookaheads
        This can be done before calculating LALR-Item-Sets, thus
        avoid computing closures repeatedly by applying the virtual
        dummy lookahead '#'.
        """
        dum = '#'
        C = [(itm, dum) for itm in I]
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
                                if b and b not in jlk: jlk.append(b)
                            if None not in G.first(X): break
                        for b in jlk:
                            jtm = G.Item(j, 0)
                            if (jtm, b) not in C: C.append((jtm, b))
            z += 1
        return C

    def calc_lalr_item_sets(G):
        Ks = [[G.Item(0, 0)]]   # Kernels
        goto = []
        spont = []
        proga = []

        # Calculate Item Sets, GOTO and progagation graph in one pass.
        i = 0
        while i < len(Ks):

            K = Ks[i]
            C = G.close_default(K)

            # Use OrderedDict to preserve order of finding
            # goto's, which should be the same order with
            # the example in textbook.
            igoto = OrderedDict()
            # Should be defaultOrderedDict
            # ispont = defaultdict(set)
            ispont = OrderedDict()
            iproga = []

            # For each possible goto relation, conclude its
            # corresponded spont/proga property (Dichotomy).
            for itm, a in C:
                # Prepare source of goto.
                if itm not in ispont:
                    ispont[itm] = set()
                # If item has goto (A -> α.Xβ)
                if not itm.ended():
                    X = itm.active()
                    jtm = itm.shifted()
                    if X not in igoto:
                        igoto[X] = []
                    if jtm not in igoto[X]:
                        igoto[X].append(jtm)
                    if a != '#':
                        # spontaneous target from Item(cr, pos)
                        ispont[itm].add(a)
                    else:
                        # progagation target from Item(cr, pos)
                        iproga.append(itm)

            # Register local goto into global goto.
            goto.append({})
            for X, J in igoto.items():
                # The Item-Sets should be treated as UNORDERED! 
                # So sort J to identify the Lists with same items,
                # otherwise these Lists are differentiated due to
                # ordering, which though strengthens the power
                # of LALR grammar, but loses LALR`s characteristics. 
                J = sorted(J, key=lambda i: (i.r, i.pos))
                if J not in Ks:
                    Ks.append(J)
                j = Ks.index(J)
                goto[i][X] = j

            spont.append(ispont)
            proga.append(iproga)

            i += 1

        # The very top of spontaneous lookahead, which is NOT included
        # in the process above, since END symbol is not covered
        # as an legal lookahead all above.
        spont[0][G.Item(0, 0)] = {grammar.END}

        # All the kernel lookahead are almost determined by the
        # "directly spontaneously one-step progagation", but
        # further progagation should be done to garantee all
        # the potential valid lookaheads.
        b = 1
        while True:
            brk = True
            for i in range(len(Ks)):
                # Update Kernel-Set[i]
                Cas = G.close_default(Ks[i])
                for itm in Ks[i]:
                    for ctm, a in Cas:
                        if a == '#':
                            lks = spont[i][itm]
                        else:
                            lks = [a]
                        for lk in lks:
                            if lk not in spont[i][ctm]:
                                # print('At', i, ' adding ', lk, ' to ', i, "'s", ctm)
                                spont[i][ctm].add(lk)
                            if not ctm.ended():
                                X = ctm.active()
                                j = goto[i][X]
                                jtm = ctm.shifted()
                                if lk not in spont[j][jtm]:
                                    # print('At', i, ' adding ', lk, ' to ', j, "'s", jtm)
                                    spont[j][jtm].add(lk)
                                    brk = False
                # Update progagation for each goto target.
                for itm in proga[i]:
                    if not itm.ended():
                        X = itm.active()
                        for lk in spont[i][itm]:
                            j = goto[i][X]
                            jtm = itm.shifted()
                            if lk not in spont[j][jtm]:
                                # print('In progagation at', i, ' adding ', a, ' to ', j, "'s", jtm)
                                spont[j][jtm].add(lk)
                                brk = False
            if brk:
                G.passes = b
                break
            else:
                b += 1

        G.Ks = Ks
        G.GOTO = goto
        G.proga = proga
        G.table = spont

        # spont has included all the non-kernel items
        # which are not necessary if spont registers only
        # the target, not the source.
        # A table representation covering only kernel items
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
                msg += '! LALR-Conflict raised:'
                msg += '\n  - in ACTION[{}]: '.format(i)
                msg += '\n    {}'.format(pp.pformat(ACTION[i], indent=4))
                msg += "\n  - conflicting action on token '{}': ".format(lk)
                msg += "\n    {{'{}': ('reduce', {})}}".format(lk, itm)
            print(msg)

        G.conflicts = conflicts
        G.ACTION = ACTION

    def parse(G, inp):
        """
        Having the ACTION table, the parsing algorithm is as follows:
        """
        trees = []
        ss = [0]

        lexer = G.tokenize(inp)

        # Parsing process.
        tok = next(lexer)
        while 1:
            i = ss[-1]
            if tok.symb not in G.ACTION[i]:
                msg =  'LALR - Ignoring syntax error by {}\n'.format(tok)
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
                    # Any need to compact singleton Tree?
                    # if len(subts) == 0:
                    #     tree = ntar
                    # elif len(subts) == 1:
                    #     tree = (ntar, subts[0])
                    # else:
                    #     tree = (ntar, subts)
                    tree = (ntar, subts)
                    trees.append(tree)
                    # New got symbol is used for shifting.
                    ss.append(G.GOTO[ss[-1]][ntar])
                # ACCEPT
                # elif act == 'accept':
                else:
                    return trees[-1]
        raise ValueError('No enough token for completing the parse. ')
        

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

    # pp.pprint(list(enumerate(GP.ACTION)))

    # No errors/problem should be raised. 
    GP.parse('id')
    GP.parse('*id')
    GP.parse("id=id")
    GP.parse('id=*id')
    GP.parse('**x=y')

    print('\n\nNow test constructed parser for errors in input.')
    print('\nTest 1: waiting to report ingorance of `p`...\n')
    p1 = GP.parse('**o p =*q')
    pp.pprint(p1)
    print('\nTest 2: waiting to report ingorance of `b` and `d`...\n')
    p2 = GP.parse('**a b =* *c d')
    pp.pprint(p2)
