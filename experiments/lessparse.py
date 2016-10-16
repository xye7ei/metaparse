import re
import pprint

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



class LexElem:

    def __init__(self, name, pattern, handler):
        self.name = name
        self.pattern = pattern
        self.handler = handler

    def __repr__(self):
        return '{}[{}][{}]'.format(
            self.__class__.__name__,
            self.name,
            repr(self.pattern))

    def try_match(self, inp, pos):
        lexeme = self.pattern
        pos1 = pos + len(lexeme)
        if inp[pos:pos1] == lexeme:
            if self.handler:
                value = self.handler(lexeme)
            else:
                value = lexeme
            return Token(pos, self.name, lexeme, value)
        else:
            return None


class RgxLexElem(LexElem):

    def __init__(self, name, pattern, handler):
        super(RgxLexElem, self).__init__(name, pattern, handler)
        self.rgx = re.compile(pattern, re.MULTILINE)

    def try_match(self, inp, pos):
        m = self.rgx.match(inp, pos=pos)
        if m:
            lexeme = m.group()
            if self.handler:
                value = self.handler(lexeme)
            else:
                value = lexeme
            return Token(pos, self.name, lexeme, value)
        else:
            return None


class Lexer(list):

    def word(self, **kw):
        for k, v in kw.items():
            lx = LexElem(k, v, None)
            self.append(lx)

    def re(self, **kw):
        for k, v in kw.items():
            lx = RgxLexElem(k, v, None)
            self.append(lx)

    def raw(self, *lits):
        for lit in lits:
            self.append(LexElem(lit, lit, None))

    def __call__(self, is_rgx=True, precedence=None, **kw):
        assert len(kw) == 1
        name, pattern = kw.popitem()
        if is_rgx:
            lex_elem = RgxLexElem(name, pattern, None)
        else:
            lex_elem = LexElem(name, pattern, None)
        self.append(lex_elem)
        def z(func):
            lex_elem.handler = func
        return z

    def tokenize(self, inp):
        pos = 0
        while pos < len(inp):
            for le in self:
                tok = le.try_match(inp, pos)
                if tok: break
            else:
                raise ValueError("No handler for unrecognized char in input: '{}'".format(inp[pos]))
            if tok.symbol != 'IGNORED':
                yield tok
            pos = tok.end


class Grammar:

    def __repr__(self):
        return pprint.pformat(self.rules)

    def __init__(self, rules, precedence={}):
        self.start = rules[0].lhs
        rules = [Rule('{}^'.format(self.start), [self.start])] + rules
        self.rules = rules
        self.nonterminals = set()
        self.symbols = set()
        for lhs, rhs in rules:
            self.nonterminals.add(lhs)
            self.symbols.update(rhs)
        self.terminals = self.symbols - self.nonterminals

        # precedence is not only specifiable for tokens, but also for
        # symbols.
        self.precedence = precedence

        # Calc NULLABLE
        self.NULLABLE = NULLABLE = set()
        while 1:
            has_new = False
            for lhs, rhs in self.rules:
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
            for lhs, rhs in self.rules:
                # Use the FIRST[rhs] to update FIRST[lhs].
                for y in rhs:
                    z = len(FIRST[lhs])
                    FIRST[lhs].update(FIRST[y])
                    z1 = len(FIRST[lhs])
                    if z1 > z:
                        has_new = True
                    if y not in NULLABLE:
                        break
            if not has_new:
                break

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

    def __getitem__(self, ir):
        """Indexing rules."""
        return self.rules[ir]

    def __call__(self, X):
        """g(X) -> [<rule-index: int>]"""
        for ir, rule in enumerate(self.rules):
            if X == rule.lhs:
                yield ir

    def closure(G, I):
        C = I[:]
        z = 0
        while z < len(C):
            (i, p) = C[z]
            if p < len(G[i].rhs):
                X = G[i].rhs[p]
                if X in G.nonterminals:
                    for j in G(X):
                        if (j, 0) not in C:
                            C.append((j, 0))
            z += 1
        return C

    def closure1_with_lookahead(G, item, a):
        C = [(item, a)]
        z = 0
        while z < len(C):
            # G[i]: i'th rule
            (i, p), a = C[z]
            if p < len(G[i].rhs):
                X = G[i].rhs[p]
                if X in G.nonterminals:
                    for j in G(X):
                        for b in G.first_of_seq(G[i].rhs[p+1:], a):
                            if ((j, 0), b) not in C:
                                C.append(((j, 0), b))
            z += 1
        return C


class LALR:

    def __init__(self):
        self.rules = []
        self.semans = []
        self.precedence = {}
        self.lexer = Lexer()

    def __enter__(self):
        return self.lexer, self.rule

    def __exit__(self, *a, **kw):
        self.make()

    @staticmethod
    def from_grammar(grammar):
        lalr = LALR()
        lalr.make(grammar)
        return lalr

    def rule(self, func):
        rule = func_to_rule(func)
        self.rules.append(rule)
        self.semans.append(func)

    def make(self, grammar=None):

        if grammar:
            self.grammar = grammar
        else:
            self.grammar = Grammar(self.rules)
        
        G = self.grammar

        # augmented grammar - top semantics
        if len(G.rules) == len(self.semans) + 1:
            self.semans.insert(0, lambda x: x)

        self.Ks = Ks = [[(0, 0)]]
        self.GOTO = GOTO = []

        # LR(0) item sets Ks and GOTO
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

        # Conclude lookahead actions
        self.ACTION = ACTION = [[] for _ in Ks]
        for i, Xto in enumerate(GOTO):
            for X, j in Xto.items():
                if X in G.terminals:
                    ACTION[i].append((X, ('shift', j)))
        for i, L in enumerate(Ls):
            K = Ks[i]
            for (r, p), l in zip(K, L):
                # ended item
                if p == len(G[r].rhs):
                    for a in l:
                        ACTION[i].append((a, ('reduce', r)))

        # Report conflicts (not for GLR)
        self.ACTION1 = ACTION1 = [{} for _ in Ks]
        for i, A in enumerate(ACTION):
            d = ACTION1[i]
            for a, (act, arg) in A:
                if a in d:
                    # Conflict resolver here!
                    act0, arg0 = d[a]
                    redu = G.rules[arg]
                    if act0 == 'shift' and act == 'reduce':
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
        G = self.grammar

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
                    for _ in G[arg].rhs:
                        sstk.pop()
                        subs.appendleft(astk.pop())

                    if interpret:
                        tree = self.semans[arg](*subs)
                    else:
                        tree = (G[arg].lhs, list(subs))

                    if token == None and arg == 0:
                        assert sstk == [0], sstk
                        assert astk == [], astk
                        yield tree

                    astk.append(tree)
                    sstk.append(GOTO[sstk[-1]][G[arg].lhs])

                    act, arg = ACTION1[sstk[-1]][look]

                sstk.append(arg)
                # Use semantic value of token whether it is by parsing
                # or interpreting.
                astk.append(token.value)

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

    @property
    def inspect_Ks(self):
        pprint.pprint([(k, [self.show_item(i, p) for i, p in K])
                       for k, K in enumerate(self.Ks)])

    @property
    def inspect_GOTO(self):
        pprint.pprint([(k, g) for k, g in enumerate(self.GOTO)])

    @property
    def inspect_lkhs(self):
        pprint.pprint([
            [(i, self.show_item(self.Ks[i][ii])),
             (j, self.show_item(self.Ks[j][jj]))]
            for (i, ii), (j, jj) in self.propa
        ])

    @property
    def inspect_propa(self):
        pprint.pprint([
            [(i, self.show_item(self.Ks[i][ii])),
             (j, self.show_item(self.Ks[j][jj]))]
            for (i, ii), (j, jj) in self.propa
        ])

    @property
    def inspect_Ls(self):
        pprint.pprint([
            (i, [(self.show_item(itm), lkhs)
                 for itm, lkhs in zip(K, self.Ls[i])])
            for i, K in enumerate(self.Ks)
        ])

    @property
    def inspect_ACTION(self):
        pprint.pprint([
            (i, self.show_itemset(i), self.ACTION[i])
            for i, K in enumerate(self.Ks)
        ])


if __name__ == '__main__':


    lxr = Lexer()
    lxr.raw('.', ',', ';')
    lxr.re(
        IDENTIFIER='[_a-zA-Z]\w*',
    )

    @lxr(NUMBER='[1-9]\d*(\.\d*)?')
    def _(val):
        return float(val)

    @lxr(IGNORED='\s+')
    def _(val):
        return None

    pp = pprint.pprint
    pp(lxr)

    ts = lxr.tokenize('   88 isGood , notBaby .   7 ')
    pp(list(t.value for t in ts))
    # assert 0

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
        RgxLexElem('IGNORED', '\s+', id_func),
        LexElem('if', 'if', id_func),
        LexElem('then', 'then', id_func),
        LexElem('else', 'else', id_func),
        RgxLexElem('expr', r'\d+', id_func),
        RgxLexElem('single', r'\w+', id_func),
    ])
    g_ite = LALR.from_grammar(Grammar(rs1, {'then': 1, 'else': 2}))
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
