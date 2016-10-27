#!/usr/bin/env python3

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
    # Use immutable.
    rhs = tuple(rhs)
    return Rule(lhs, rhs)


def identity(x):
    return x


class Lexer(object):

    class Error(Exception):
        pass

    def __init__(self, lex2pats=None, handlers=None):
        self.lex2pats = lex2pats if lex2pats else []
        self.handlers = handlers if handlers else []
        self.precedence = {}

    def __call__(self, **kw):
        assert ('p' in kw and len(kw) == 2) or len(kw) == 1
        prece = kw.pop('p') if 'p' in kw else None
        name, pattern = kw.popitem()
        if prece: self.precedence[name] = prece
        self.lex2pats.append((name, re.compile(pattern)))
        self.handlers.append(None)
        assert len(self.lex2pats) == len(self.handlers)
        def z(func):
            'Swap the last handler with the decorated function.'
            self.handlers[-1] = func
        return z

    def __repr__(self):
        return pprint.pformat(self.lex2pats)

    def register(self, name, pattern, handler=None, precedence=None):
        self.lex2pats.append((name, re.compile(pattern)))
        self.handlers.append(handler)
        if precedence != None:
            self.precedence[name] = precedence

    def more(self, **kw):
        for k, v in kw.items():
            self.register(k, v)

    def tokenize(self, inp):
        lex2pats = self.lex2pats
        handlers = self.handlers
        pos = 0
        while pos < len(inp):
            match = None
            name = None
            handler = None
            for (nm, rgx), hdl in zip(lex2pats, handlers):
                match = rgx.match(inp, pos=pos)
                if match:
                    name = nm
                    handler = hdl
                    break
            else:
                raise Lexer.Error(
                    "No pattern for unrecognized: {}th char in input: '{}'\n"
                    .format(pos, inp[pos]))
            lxm = match.group()
            if name == 'IGNORED':
                # IGNORED is not associated with any handler.
                pass
            elif name == 'ERROR':
                # ERROR must have a handler, whilst not yielded as a token.
                assert handler
                handler(lxm)
            else:
                val = handler(lxm) if handler else lxm
                yield Token(pos, name, lxm, val)
            pos = match.end()


class Grammar(object):

    def __init__(self, rules, precedence=None):

        if not precedence:
            precedence = {}

        # Augmented grammar with singleton/non-alternated start-rule.
        self.start = rules[0].lhs
        self.rules = rules

        # Conclude nonterminals/terminals.
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
        # Collect unreachable nonterminal from start symbol.
        reachable = {self.start}
        while 1:
            news = set()
            for X in reachable:
                for j in self.group[X]:
                    for Y in self.rules[j].rhs:
                        if Y in self.nonterminals:
                            if Y not in reachable:
                                news.add(Y)
            if news: reachable.update(news)
            else: break
        self.unreachable = self.nonterminals - reachable

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
                for Y in rhs:
                    for a in FIRST[Y]:
                        if a not in FIRST[lhs]:
                            FIRST[lhs].add(a)
                            has_new = True
                    if Y not in NULLABLE:
                        break
            if not has_new:
                break

    def __repr__(self):
        return pprint.pformat(self.rules)

    def first(self, X):
        if X in self.FIRST:
            return self.FIRST[X]
        else:
            return {X}

    def first_of_seq(self, seq, tail):
        assert tail != 'EPSILON'
        s = set()
        # `for-else` structure: do-and-find sth, if not found, run `else`.
        for Y in seq:
            s.update(self.first(Y))
            if Y not in self.NULLABLE:
                break
        else:
            # `else` is executed only when `for` is not broken out.
            s.add(tail)
        s.discard('EPSILON')
        return s

    def closure(self, I):
        """Naive closure algorithm on item set."""
        G = self
        C = I[:]
        z = 0
        while z < len(C):
            (i, p) = C[z]
            if p < len(G.rules[i].rhs):
                X = G.rules[i].rhs[p]
                if X in G.nonterminals:
                    for j in G.group[X]:
                        if (j, 0) not in C:
                            C.append((j, 0))
            z += 1
        return C

    def closure1_with_lookahead(self, item, a):
        """Lookahead closure algorithm on singleton item set."""
        G = self
        C = [(item, a)]
        z = 0
        while z < len(C):
            (i, p), a = C[z]
            if p < len(G.rules[i].rhs):
                X = G.rules[i].rhs[p]
                if X in G.nonterminals:
                    for j in G.group[X]:
                        for b in G.first_of_seq(G.rules[i].rhs[p+1:], a):
                            if ((j, 0), b) not in C:
                                C.append(((j, 0), b))
            z += 1
        return C

    class meta(type):

        class Reader(list):

            def __getitem__(self, k):
                raise KeyError()

            def __setitem__(self, k, v):
                if callable(v):
                    self.append(func_to_rule(v))
                else:
                    pass

        def __prepare__(mcls, *a, **kw):
            return Grammar.meta.Reader()

        def __new__(mcls, n, b, r):
            return Grammar(list(r))


def augment(rules, semans):
    assert len(rules) == len(semans)
    start = rules[0].lhs
    rules = [Rule(start+'^', (start,))] + rules
    semans = [identity] + semans
    return rules, semans


class LALR(object):

    class Error(Exception):
        pass

    def __init__(self, lexer=None, rules=None, precedence=None):
        self.rules = rules if rules else []
        self.precedence = precedence if precedence else {}
        self.lexer = lexer if lexer else Lexer()
        self.semans = []

    def rule(self, func):
        rule = func_to_rule(func)
        self.rules.append(rule)
        self.semans.append(func)

    def make(self):

        # Augmented lexer - ignoring spaces by default.
        lexes = {lex for lex, _ in self.lexer.lex2pats}
        if 'IGNORED' not in lexes:
            self.lexer.register('IGNORED', r'\s+')

        # Augmented grammar - top semantics
        self.rules, self.semans = augment(self.rules, self.semans)
        # rules = self.rules
        # semans = self.semans

        # Propagate precedence from lexer.
        if self.lexer.precedence:
            self.precedence.update(self.lexer.precedence)

        # Prepare Grammar object to use closure algorithms.
        G = Grammar(self.rules, self.precedence)

        # if 'ERROR' not in self.lexer.handler:
        #     warnings.warn(
        #         "No ERROR handler available. "
        #         "Lexer will fail when reading unrecognized character.")

        # Check coverage of Lexer.
        # - Each terminal should have its corresponding lexical pattern.
        for r, rule in enumerate(G.rules):
            for y in rule.rhs:
                if y in G.terminals and y not in lexes:
                    msg = ('No lexical pattern provided for terminal symbol: {}\n'
                           '- in {}th rule {}\n'
                    ).format(y, r, rule)
                    seman = self.semans[r]
                    import traceback
                    trc = traceback.format_list([
                        (seman.__code__.co_filename,
                         seman.__code__.co_firstlineno,
                         seman.__name__,
                         '')])[0]
                    trc_msg = ('- with helping traceback (if available): \n'
                               '{}\n').format(trc)
                    lex_msg = str(self.lexer)
                    raise LALR.Error(msg + trc_msg + lex_msg)
        # Report soundness of grammar (unreachable, loops, etc).
        for X in G.unreachable:
            for i in G.group[X]:
                seman = self.semans[i]
                import traceback
                trc = traceback.format_list([
                    (seman.__code__.co_filename,
                     seman.__code__.co_firstlineno,
                     seman.__name__,
                     '')])[0]
                msg = ('There are unreachable nonterminals: {}.\n'
                       '- with helping traceback: \n{}\n'
                ).format(G.unreachable, trc)
                # warnings.warn(msg)
                raise LALR.Error(msg)

        # Kernel sets and corresponding GOTO
        self.Ks = Ks = [[(0, 0)]]
        self.GOTO = GOTO = []

        # Make LR(0) kernel sets Ks and GOTO, incrementally.
        i = 0
        while i < len(Ks):
            I = Ks[i]
            igotoset = odict()
            for (nk, p) in G.closure(I):
                if p < len(G.rules[nk].rhs):
                    X = G.rules[nk].rhs[p]
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
                    if p < len(G.rules[nk].rhs):
                        # actor
                        X = G.rules[nk].rhs[p]
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
                    if q == len(G.rules[c].rhs):
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
            # Try add (act, arg) into d.
            for a, (act, arg) in A:
                # It is assured that 'shift' is added earlier than 'reduce'
                if a in d:
                    # Conflict resolver here!
                    act0, arg0 = d[a]
                    # if {act0, act} == {'shift', 'reduce'}:
                    if {act0, act} == {'shift', 'reduce'}:
                        if act0 == 'reduce':
                            s, s_i = act, arg
                            r, r_r = act0, arg0
                        else:
                            s, s_i = act0, arg0
                            r, r_r = act, arg
                        redu = G.rules[r_r]
                        if a in G.precedence:
                            if len(redu.rhs) > 1 and redu.rhs[-2] in G.precedence:
                                lft = redu.rhs[-2]
                                rgt = a
                                if G.precedence[lft] >= G.precedence[rgt]:
                                    d[a] = (r, r_r)
                                else:
                                    d[a] = (s, s_i)
                                continue
                    # Unable to resolve
                    msg = ("\n"
                           "Handling item set: \n" "{}\n"
                           "Conflict on lookahead: {} \n"
                           "- {}\n" "- {}\n"
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
        return rtn.send(None)

    def interpret(self, inp):
        assert self.semans, 'Must have semantics to interpret.'
        return self.parse(inp, True)

    def dumps(self):
        'Dump this parser instance to readable Python code string.'

        tar = odict()

        tar['lex2pats'] = [
            (nm, rgx.pattern)
            for nm, rgx in self.lexer.lex2pats
        ]
        tar['handlers'] = [
            marshal.dumps(h.__code__) if h else None
            for h in self.lexer.handlers
        ]

        tar['rules'] = [tuple(rl) for rl in self.rules]
        tar['ACTION1'] = self.ACTION1
        tar['GOTO'] = self.GOTO

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
        'Load a dumped code string and make a usable parse instance.'
        ctx = {}
        exec(src, env, ctx)
        lex2pats = [(nm, re.compile(pat))
                 for nm, pat in ctx.pop('lex2pats')]
        handlers = [
            types.FunctionType(marshal.loads(co), env) if co else None
            for co in ctx.pop('handlers')
        ]

        p = LALR(Lexer(lex2pats, handlers))
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
        lhs, rhs = self.rules[i]
        return '({} = {}.{})'.format(lhs,
                                     ' '.join(rhs[:p]),
                                     ' '.join(rhs[p:]))

    def show_itemset(self, i):
        return ([self.show_item(tm) for tm in self.Ks[i]])

    def show_action(self, action):
        act, arg = action
        return (act, self.show_itemset(arg) if act == 'shift' else self.rules[arg])

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

    # Various style of declaration.
    def __getitem__(self, k):
        raise KeyError()

    def __setitem__(self, k, v):

        if k == '__doc__':
            self.__doc__ = v

        elif k.startswith('__') and k.endswith('__'):
            pass

        elif isinstance(v, str):
            self.lexer.add(k, v)

        elif isinstance(v, tuple):
            assert len(v) == 2
            pat, prece = v
            self.lexer.add(k, pat)
            if prece in self.precedence:
                raise ValueError(
                    'Repeated specifying the precedence of symbol: {}'.format(k))
            else:
                self.precedence[k] = prece

        elif callable(v):
            parlist = v.__code__.co_varnames[:v.__code__.co_argcount]
            if len(parlist) == 1 and parlist[0] in ('lex', 'LEX'):
                for _, pat in v.__annotations__.items():
                    self.lexer.add(k, pat, v)
            else:
                self.rule(v)

    def __enter__(self):
        return self.lexer, self.rule

    def __exit__(self, *a, **kw):
        self.make()

    @staticmethod
    def verbose(func_def):
        assert func_def.__code__.co_argcount == 2
        p = LALR()
        func_def(p.lexer, p.rule)
        p.make()
        return p

    class meta(type):

        def __prepare__(mcls, *a, **kw):
            return LALR()

        def __new__(mcls, m, bs, p):
            p.make()
            return p


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

    def id_func(a):
        return a

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
            self.assertEqual(set(g.NULLABLE), {'S', 'A', 'B', 'C', 'D', 'E'})

    unittest.main()
