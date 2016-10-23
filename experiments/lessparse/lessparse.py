import re
import pprint
import warnings
import marshal, types

from pprint import pformat
from textwrap import indent, dedent

from collections import deque
from collections import namedtuple
from collections import OrderedDict as odict


Token = namedtuple('Token', 'pos symbol lexeme value')
Token.__repr__ = lambda self: "({}: {})".format(repr(self.symbol), repr(self.value))
Token.end = property(lambda self: self.pos + len(self.lexeme))


Rule = namedtuple('Rule', 'lhs rhs')
Rule.__repr__ = lambda self: '({} = {})'.format(self.lhs, ' '.join(self.rhs))


def func_to_rule(func):
    'Construct a rule object from a function/method. '
    lhs = func.__name__
    rhs = []
    ac = func.__code__.co_argcount
    vs = func.__code__.co_varnames
    for x in vs[:ac]:
        # Cut tailing digital subscript like xxx_4.
        s = re.search(r'_(\d+)$', x)
        if s:
            x = x[:s.start()]
        rhs.append(x)
    # Make it immutable.
    rhs = tuple(rhs)
    return Rule(lhs, rhs)


def identity(x):
    return x


class Lexer(object):

    def __init__(self, lexes=None, handler=None):
        self.lexes = lexes if lexes else []
        self.handler = handler if handler else {}

    def __call__(self, **kw):
        assert len(kw) == 1
        name, pattern = kw.popitem()
        self.lexes.append((name, re.compile(pattern)))
        def z(func):
            self.handler[name] = func
        return z

    def more(self, **kw):
        for k, v in kw.items():
            self(**{k: v})

    def tokenize(self, inp):
        lexes = self.lexes
        handler = self.handler
        pos = 0
        while pos < len(inp):
            name = None
            m = None
            for nm, rgx in lexes:
                name = nm
                m = rgx.match(inp, pos=pos)
                if m: break
            else:
                raise ValueError("No pattern for unrecognized char in input: '{}'".format(inp[pos]))
            lxm = m.group()
            if name != 'IGNORED':
                if name in handler:
                    value = handler[name](lxm)
                else:
                    value = lxm
                yield Token(pos, name, lxm, value)
            pos = m.end()


class Grammar(object):

    def __init__(self, rules, precedence={}):

        # Augmented grammar with singleton/non-alternated start-rule.
        self.start = rules[0].lhs
        self.rules = rules

        # 
        self.nonterminals = set()
        self.symbols = set()
        for lhs, rhs in rules:
            self.nonterminals.add(lhs)
            self.symbols.update(rhs)
        self.terminals = self.symbols - self.nonterminals

        # Group by LHS
        self.group = {nt: [] for nt in self.nonterminals}
        for i, (lhs, rhs) in enumerate(rules):
            self.group[lhs].append(i)

        # precedence is not only specifiable for tokens, but also for
        # symbols.
        self.precedence = precedence

        # Calc NULLABLE
        self.NULLABLE = NULLABLE = set()
        while 1:
            has_new = False
            for lhs, rhs in rules:
                if all(x in NULLABLE for x in rhs):
                    if lhs not in NULLABLE:
                        NULLABLE.add(lhs)
                        has_new = True
            if not has_new:
                break

        # Calc FIRST
        self.FIRST = FIRST = {}
        for t in self.terminals:
            FIRST[t] = {t}
        for nt in self.nonterminals:
            FIRST[nt] = set()
            if nt in NULLABLE:
                FIRST[nt].add('EPSILON')
        while 1:
            has_new = False
            for lhs, rhs in rules:
                # Use the FIRST[rhs] to update FIRST[lhs].
                for y in rhs:
                    # z = len(FIRST[lhs])
                    # FIRST[lhs].update(FIRST[y])
                    # z1 = len(FIRST[lhs])
                    # if z1 > z:
                    #     has_new = True
                    for a in FIRST[y]:
                        if a not in FIRST[lhs]:
                            FIRST[lhs].add(a)
                            has_new = True
                    if y not in NULLABLE:
                        break
            if not has_new:
                break

    def __repr__(self):
        return pprint.pformat(self.rules)

    def __getitem__(self, ir):
        """Indexing rules."""
        return self.rules[ir]

    def first(self, X):
        if X in self.FIRST:
            return self.FIRST[X]
        else:
            return {X}

    def first_of_seq(self, seq, tail):
        s = set()
        # `for-else` structure: do-and-find sth, if not found, run `else`.
        for Y in seq:
            s.update(self.first(Y))
            if Y not in self.NULLABLE:
                break
        else:
            # `else` is executed only when `for` is not broken out.
            s.add(tail)
        return s

    def closure(G, I):
        """Naive closure algorithm on item set."""
        C = I[:]
        z = 0
        while z < len(C):
            (i, p) = C[z]
            if p < len(G[i].rhs):
                X = G[i].rhs[p]
                if X in G.nonterminals:
                    for j in G.group[X]:
                        if (j, 0) not in C:
                            C.append((j, 0))
            z += 1
        return C

    def closure1_with_lookahead(G, item, a):
        """Lookahead closure algorithm on singleton item set."""
        C = [(item, a)]
        z = 0
        while z < len(C):
            # G[i]: i'th rule
            (i, p), a = C[z]
            if p < len(G[i].rhs):
                X = G[i].rhs[p]
                if X in G.nonterminals:
                    for j in G.group[X]:
                        for b in G.first_of_seq(G[i].rhs[p+1:], a):
                            if ((j, 0), b) not in C:
                                C.append(((j, 0), b))
            z += 1
        return C


def augment(rules, semans):
    assert len(rules) == len(semans)
    start = rules[0].lhs
    rules = [Rule(start+'^', (start,))] + rules
    semans = [identity] + semans
    return rules, semans


class LALR(object):

    def __init__(self, lexer=None, rules=None, precedence=None):
        self.rules = rules if rules else []
        self.precedence = precedence if precedence else {}
        self.lexer = lexer if lexer else Lexer()
        self.semans = []

    def __enter__(self):
        return self.lexer, self.rule

    def __exit__(self, *a, **kw):
        self.make()

    def rule(self, func):
        rule = func_to_rule(func)
        self.rules.append(rule)
        self.semans.append(func)

    def make(self):
        
        # augmented grammar - top semantics
        self.rules, self.semans = augment(self.rules, self.semans)
        ruls = self.rules
        semans = self.semans

        G = Grammar(self.rules, self.precedence)

        self.Ks = Ks = [[(0, 0)]]
        self.GOTO = GOTO = []

        # Make LR(0) kernel sets Ks and GOTO, incrementally.
        i = 0
        while i < len(Ks):
            I = Ks[i]
            igotoset = odict()
            for (nk, p) in G.closure(I):
                if p < len(G[nk].rhs):
                    X = G[nk].rhs[p]
                    if X not in igotoset:
                        igotoset[X] = []
                    if (nk, p+1) not in igotoset[X]:
                        igotoset[X].append((nk, p+1)) # shifted item (nk, p+1)
            igoto = {}
            for X, J in igotoset.items():
                J.sort()
                if J in Ks:
                    igoto[X] = Ks.index(J)
                else:
                    igoto[X] = len(Ks)
                    Ks.append(J)
            GOTO.append(igoto)
            i += 1

        # Lookahead set corresponding to item set
        self.Ls = Ls = [[set() for _ in K] for K in Ks]

        Ls[0][0] = {'\x03'}
        # Ls[0][0] = {'$'}

        DUMMY = '\x00'
        propa = []
        for i, K in enumerate(Ks):
            for ii, itm in enumerate(K):
                C = G.closure1_with_lookahead(itm, DUMMY)
                # for each non-kernel nk
                for (nk, p), a in C:
                    # active
                    if p < len(G[nk].rhs):
                        # actor
                        X = G[nk].rhs[p]
                        # target item
                        j = GOTO[i][X]
                        jj = Ks[j].index((nk, p+1))
                        # spontaneous
                        if a != DUMMY:
                            Ls[j][jj].add(a)
                        # propagated
                        else:
                            propa.append((
                                # from K[i], ii'th item
                                (i, ii),
                                # to K[j], jj'th item
                                (j, jj),
                            ))
                    # @@@ ended item?
                    # @@@ conclude kernel/nonkernel 'reduce' (nk, p) in Ks[i] on lookahead a?
                    # @@@ BUT here a may be dummy!
                    # @@@ the item to be reduced should share set of lookaheads of kernel item
                    # @@@ BUT this set is yet to be accomplished.

        # Propagation till fix-point
        self.propa = propa
        while 1:
            has_new = False
            for (i, ii), (j, jj) in propa:
                for a in Ls[i][ii]:
                    if a not in Ls[j][jj]:
                        Ls[j][jj].add(a)
                        has_new = True
            if not has_new:
                break

        # Conclude lookahead actions allowing conflicts on identical
        # lookaheads.
        self.ACTION = ACTION = [set() for _ in Ks]
        for i, Xto in enumerate(GOTO):
            for X, j in Xto.items():
                if X in G.terminals:
                    ACTION[i].add((X, ('shift', j)))
        for i, L in enumerate(Ls):
            K = Ks[i]
            for k, l in zip(K, L):
                for (c, q), b in G.closure1_with_lookahead(k, DUMMY):
                    # IMPORTANT: kernel/non-kernels which are ended!
                    if q == len(G[c].rhs):
                        # spontaneous reduction
                        if b != DUMMY:
                            ACTION[i].add((b, ('reduce', c)))
                        # propagated from lookaheads of kernel item being closed
                        else:
                            for a in l:
                                ACTION[i].add((a, ('reduce', c)))

        # Resolve conflicts (not for GLR)
        self.ACTION1 = ACTION1 = [{} for _ in Ks]
        for i, A in enumerate(ACTION):
            d = ACTION1[i]
            for a, (act, arg) in A:
                if a in d:
                    # Conflict resolver here!
                    act0, arg0 = d[a]
                    redu = G.rules[arg]
                    if {act0, act} == {'shift', 'reduce'}:
                        if a in G.precedence:
                            if len(redu.rhs) > 1 and redu.rhs[-2] in G.precedence:
                                lft = redu.rhs[-2]
                                rgt = a
                                if G.precedence[lft] >= G.precedence[rgt]:
                                    d[a] = (act, arg)
                                else:
                                    d[a] = (act0, arg0)
                                continue
                    # Unable to resolve
                    msg = ("\n"
                           "Handling item set: \n"
                           "{}\n"
                           "Conflict on lookahead:: {} \n"
                           "- {}\n"
                           "- {}\n"
                    ).format(
                        self.show_itemset(i),
                        a,
                        self.show_action(d[a]),
                        self.show_action((act, arg)),
                    )
                    raise ValueError(msg)
                else:
                    d[a] = (act, arg)

    def prepare(self, interpret=True):
        sstk = [0]              # state stack
        astk = []               # arg stack
        ACTION1 = self.ACTION1
        GOTO = self.GOTO
        rules = self.rules

        while 1:
            token = (yield)
            # token = (yield)
            # inp = (yield astk)
            # for token in self.lexer.tokenize(inp):
            if token == None:   # Finish
                # assert sstk == [0], sstk
                # yield astk[0]
                look = '\x03'
            else:
                look = token.symbol

            if look in ACTION1[sstk[-1]]:
                act, arg = ACTION1[sstk[-1]][look]
                # Reduce until shift happens!
                while act == 'reduce':
                    subs = deque()
                    for _ in rules[arg].rhs:
                        sstk.pop()
                        subs.appendleft(astk.pop())

                    if interpret:
                        tree = self.semans[arg](*subs)
                    else:
                        tree = (rules[arg].lhs, list(subs))

                    if token == None and arg == 0:
                        assert sstk == [0], sstk
                        assert astk == [], astk
                        yield tree

                    astk.append(tree)
                    sstk.append(GOTO[sstk[-1]][rules[arg].lhs])

                    act, arg = ACTION1[sstk[-1]][look]

                sstk.append(arg)
                # Use semantic value of token whether it is by parsing
                # or interpreting.
                astk.append(token.value)
            else:
                msg = ('Unexpected lookahead symbol: {}\n'
                       'current ACTION: \n'
                       '{}').format(look, ACTION1[sstk[-1]])
                warnings.warn(msg)

    def parse(self, inp, interpret=False):
        rtn = self.prepare(interpret)
        next(rtn)
        for token in self.lexer.tokenize(inp):
            rtn.send(token)
        else:
            return rtn.send(None)

    def interpret(self, inp):
        assert self.semans, 'Must have semantics to interpret.'
        return self.parse(inp, True)

    def dumps(self):
        tar = odict()

        tar['lexes'] = [
            (nm, rgx.pattern)
            for nm, rgx in self.lexer.lexes
        ]
        tar['rules'] = [tuple(rl) for rl in self.rules]
        tar['ACTION1'] = self.ACTION1
        tar['GOTO'] = self.GOTO

        tar['lexer_handler_code'] = {
            nm: marshal.dumps(h.__code__)
            for nm, h in self.lexer.handler.items()
        }
        tar['semans'] = [
            marshal.dumps(f.__code__)
            for f in self.semans
        ]

        return '\n'.join(
            '{} = \\\n{}\n'.format(k, indent(pformat(v), '    '))
             for k, v in tar.items()
        )

    @staticmethod
    def loads(src, env=globals()):
        ctx = {}
        exec(src, env, ctx)
        lexes = [(nm, re.compile(pat))
                 for nm, pat in ctx.pop('lexes')]
        handler = {nm: types.FunctionType(marshal.loads(co), env)
                   for nm, co in ctx.pop('lexer_handler_code').items()}
        p = LALR()
        p.lexer = Lexer(lexes, handler)
        p.rules = [Rule(*rl) for rl in ctx.pop('rules')]
        p.ACTION1 = ctx.pop('ACTION1')
        p.GOTO = ctx.pop('GOTO')
        p.semans = [
            types.FunctionType(marshal.loads(co), env)
            for co in ctx.pop('semans')
        ]

        return p

    # Helper for easy reading states.
    def show_item(self, item):
        i, p = item
        lhs, rhs = self.grammar[i]
        return '({} = {}.{})'.format(lhs,
                                     ' '.join(rhs[:p]),
                                     ' '.join(rhs[p:]))

    def show_itemset(self, i):
        return ([self.show_item(tm) for tm in self.Ks[i]])

    def show_action(self, action):
        act, arg = action
        return (act, self.show_itemset(arg) if act == 'shift' else self.grammar.rules[arg])

    def inspect_Ks(self):
        pprint.pprint([(k, [self.show_item(itm) for itm in K])
                       for k, K in enumerate(self.Ks)])

    def inspect_lkhs(self):
        pprint.pprint([
            [(i, self.show_item(self.Ks[i][ii])),
             (j, self.show_item(self.Ks[j][jj]))]
            for (i, ii), (j, jj) in self.propa
        ])

    def inspect_propa(self):
        pprint.pprint([
            [(i, self.show_item(self.Ks[i][ii])),
             (j, self.show_item(self.Ks[j][jj]))]
            for (i, ii), (j, jj) in self.propa
        ])

    def inspect_Ls(self):
        pprint.pprint([
            (i, [(self.show_item(itm), lkhs)
                 for itm, lkhs in zip(K, self.Ls[i])])
            for i, K in enumerate(self.Ks)
        ])

    def inspect_ACTION(self):
        pprint.pprint([
            (i, self.show_itemset(i), self.ACTION[i])
            for i, K in enumerate(self.Ks)
        ])

    def inspect_GOTO(self):
        pprint.pprint([
            (i, self.show_itemset(i), self.GOTO[i])
            for i, K in enumerate(self.Ks)
        ])


if __name__ == '__main__':


    rs = ([
        Rule('S', ('A', 'B', 'C')),
        Rule('S', ('D',)),
        Rule('A', ('a', 'A')),
        Rule('A', ()),
        Rule('B', ('B', 'b')),
        Rule('B', ()),
        Rule('C', ('c',)),
        Rule('C', ('D',)),
        Rule('D', ('d', 'D')),
        Rule('D', ('E',)),
        Rule('E', ('D',)),
        Rule('E', ('B',)),
    ])
    g = Grammar(rs)

    rs1 = [
        Rule('expr', ['expr', '+', 'term']),
        Rule('expr', ['term']),
        Rule('term', ['term', '*', 'factor']),
        Rule('term', ['factor']),
        Rule('factor', ['ID']),
        Rule('factor', ['(', 'expr', ')']),
    ]
    e = Grammar(rs1)

    rs1 = [
        Rule('S', ['L', '=', 'R']),
        Rule('S', ['R']),
        Rule('L', ['*', 'R']),
        Rule('L', ['id']),
        Rule('R', ['L']),
    ]

    rs1 = [
        Rule('stmt', ['if', 'expr', 'then', 'stmt']),
        Rule('stmt', ['if', 'expr', 'then', 'stmt', 'else', 'stmt']),
        Rule('stmt', ['single']),
    ]

    # (l.inspect_Ks)
    # (l.inspect_GOTO)
    # (l.inspect_propa)
    # (l.inspect_Ls)
    # (l.inspect_ACTION)

    # l = LALR(Grammar(rs1, {'then': 1}))

    def id_func(a):
        return a

    lx_ite = Lexer([
        ('IGNORED', r'\s+'),
        ('if', 'if'),
        ('then', 'then'),
        ('else', 'else'),
        ('expr', r'\d+'),
        ('single', r'\w+'),
    ])
    g_ite = LALR(lx_ite, rs, {'then': 1, 'else': 2})
    g_ite.lexer = lx_ite

    f = g_ite.prepare(False)
    tks = list(lx_ite.tokenize('if 123 then abc'))
    pprint = pprint.pprint
    next(f)
    pprint(f.send(tks[0]))
    pprint(f.send(tks[1]))
    pprint(f.send(tks[2]))
    pprint(f.send(tks[3]))
    pprint(f.send(None))

    import unittest
    class TestGrammar(unittest.TestCase):
        def test_first_0(self):
            self.assertEqual(g.FIRST['S'], {'a', 'b', 'c', 'd', 'EPSILON'})
            self.assertEqual(g.FIRST['E'], {'b', 'd', 'EPSILON'})
        def test_first_1(self):
            self.assertEqual(e.FIRST['expr'], {'ID', '('})
            self.assertEqual(e.FIRST['term'], {'ID', '('})
            self.assertEqual(e.FIRST['factor'], {'ID', '('})
        def test_nullalbe(self):
            self.assertEqual(set(g.NULLABLE), {'S^', 'S', 'A', 'B', 'C', 'D', 'E'})

    unittest.main()
