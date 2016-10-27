# -*- coding: utf-8 -*-
"""``metaparse``

A extensive tool for handy and powerful instant parsing.

Usage:

    Just declare a Python class with several methods to make a
    handy compiler.


Example:

    Suppose we have a grammar like

        expr -> NUM
        expr -> expr + expr
        expr -> expr * expr
        expr -> expr ** expr

    with some adaptation, a compiler can then be declared as

.. code:: python

    from metaparse import LALR

    class pCalc(metaclass=LALR.meta):

        IGNORED = r' '
        IGNORED = r'\\t'        # alternative patterns

        NUM = r'[0-9]+'         # re-pattern
        def NUM(lex):           # lexical handler
            return int(lex)

        POW = r'\*\*', 3        # precedence (greater the higher)
        MUL = r'\*'  , 2
        ADD = r'\+'  , 1

        def expr(NUM):
            return NUM

        def expr(expr_1, ADD, expr_2):   # subscripts 
            return expr_1 + expr_2

        def expr(expr, MUL, expr_1):     # fewer subscripts
            return expr * expr_1

        def expr(expr, POW, expr_1):
            return expr ** expr_1


    # Interpret
    >>> pCalc.interpret("1 + 2 * 3 ** 4 + 5")
    168

    # Parse
    >>> pCalc.parse("1 + 2 * 3")
    (expr,
     [(expr, [(NUM: '1')@[0:1]]),
      (ADD: '+')@[2:3],
      (expr,
       [(expr, [(NUM: '2')@[4:5]]),
        (MUL: '*')@[6:7],
        (expr, [(NUM: '3')@[8:9]])])])

TODO:
    - Online-parsing


INFO:
    Author:  ShellayLee
    Mail:    shellaylee@hotmail.com
    License: MIT


.. _Project site:
    https://github.com/Shellay/metaparse

"""

import re
import warnings
import inspect
import textwrap
import ast
import pprint as pp
import json, marshal, pickle
import types

from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple
from collections import deque
from collections import Iterable


class GrammarError(Exception):
    """Specifies exceptions raised when constructing a grammar."""


class LexerError(Exception):
    """Error for tokenization."""


class ParserError(Exception):
    """Specifies exceptions raised when error occured during parsing."""


# Special lexical element and lexemes
END = 'END'
END_PATTERN_DEFAULT = r'\Z'

IGNORED = 'IGNORED'
IGNORED_PATTERN_DEFAULT = r'[ \t\n]'

ERROR = 'ERROR'
ERROR_PATTERN_DEFAULT = r'.'

PRECEDENCE_DEFAULT = 1


class Symbol(str):

    """Symbol is a subclass of :str: with unquoted representation. It is
    only refered for cleaner presentation.

    """

    def __repr__(self):
        return self


DUMMY   = Symbol('\0')
EPSILON = Symbol('EPSILON')

START   = ('START')
PREDICT = ('PREDICT')
SHIFT   = ('SHIFT')
REDUCE  = ('REDUCE')
ACCEPT  = ('ACCEPT')


def id_func(x):
    return x


class Token(object):

    """Token object is the lexical element of a grammar, which also
    includes the lexeme's position and literal value.

    """

    def __init__(self, at, symbol, lexeme, value):
        self.at = at
        self.symbol = symbol
        self.lexeme = lexeme
        self.value = value

    def __repr__(self):
        return '({}: {})@[{}:{}]'.format(
            self.symbol,
            repr(self.lexeme),
            self.at,
            self.at + len(self.lexeme))

    def __eq__(self, other):
        return self.symbol == other.symbol

    def __iter__(self):
        yield self.at
        yield self.symbol
        yield self.lexeme
        # yield self.value

    def is_END(self):
        return self.symbol == END

    @property
    def start(self):
        return self.at

    @property
    def end(self):
        return self.at + len(self.value)


class Rule(object):

    """Rule object has the form (LHS -> RHS).  It can be constructed by
    a function, taking its name as LHS and parameter list as RHS. The
    function itself is treated as the semantical behavior of this rule.

    The equality is needed mainly for detecting rule duplication
    during declaring grammar objects.

    Note: a rule is associated with some semantics. More advancedly,
    pre-order semantics and post-order semantics may be prepared, with
    the former executed when the rule is chosen by the parser
    algorithm, whilst the latter executed when the rule is completed.

    """

    def __init__(self, lhs, rhs):
        # Make use of annotations?
        # self.anno = func.__annotations__
        self.lhs = lhs
        self.rhs = rhs

    def __eq__(self, other):
        """Equality of Rule object relies only upon LHS and RHS, not
        including semantics! """
        if isinstance(other, Rule):
            return (self.lhs == other.lhs) and (self.rhs == other.rhs)
        else:
            return False

    def __repr__(self):
        """There are alternative representations for "produces" like '->' or
        '::='. Here '=' is used for simplicity.

        """
        return '({} = {})'.format(self.lhs, ' '.join(self.rhs))

    def __iter__(self):
        yield self.lhs
        yield self.rhs

    @staticmethod
    def read(func, early=None):
        'Construct a rule object from a function/method. '
        lhs = func.__name__
        rhs = []
        # `inspect.Signature` only works in Python 3
        # for x in inspect.signature(func).parameters:
        # for x in inspect.getargspec(func).args:
        # Or without using `inspect`
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

    @property
    def size(self):
        return len(self.rhs)

    # FIXME: where to cache source info
    @property
    def src_info(self):
        """Retrieves source information of this rule's definition for helpful
        Traceback information.

        """
        co = self.seman.__code__
        info = '  File "{}", line {}, in {}\n'.format(
            co.co_filename, co.co_firstlineno, self.seman.__module__)
        return info


class Item(object):

    """Item contains a pointer to a rule list, a index of rule within that
    list and a position indicating current active symbol (namely
    actor) in this Item.

    """

    def __init__(self, rules, r, pos):
        self.rules = rules
        self.r = r
        self.pos = pos

    def __repr__(s):
        rule = s.rules[s.r]
        lhs = rule.lhs
        rhs1 = ' '.join(rule.rhs[:s.pos])
        rhs2 = ' '.join(rule.rhs[s.pos:])
        return '({} = {}.{})'.format(lhs, rhs1, rhs2)

    def __eq__(self, x):
        "Only compare the indices."
        assert isinstance(x, Item), x
        return self.r == x.r and self.pos == x.pos

    def __hash__(self):
        return hash((self.r, self.pos))

    def __lt__(self, other):
        "Support sorting of item objects."
        return self.index_pair < other.index_pair

    @property
    def rule(s):
        return s.rules[s.r]

    def ended(s):
        return s.pos == len(s.rules[s.r].rhs)

    @property
    def rest(s):
        return s.rules[s.r].rhs[s.pos:]

    @property
    def look_over(s):
        return s.rules[s.r].rhs[s.pos+1:]

    @property
    def prev(s):
        return s.rule.rhs[s.pos-1]

    @property
    def actor(s):
        return s.rules[s.r].rhs[s.pos]

    @property
    def shifted(s):
        return Item(s.rules, s.r, s.pos+1)

    @property
    def unshifted(s):
        return s.rules[s.r].rhs[s.pos-1]

    @property
    def size(s):
        return len(s.rules[s.r].rhs)

    @property
    def target(s):
        return s.rules[s.r].lhs

    @property
    def index_pair(s):
        return (s.r, s.pos)


# Component for graph structured stacks and prediction trees.
Node = namedtuple('Node', 'value next')
Node.__repr__ \
    = lambda s: tuple.__repr__((Symbol(s[0]), s[1]))
Node.to_list \
    = lambda s: [s.value] + (s.next.to_list() if isinstance(s.next, Node) else [s.next])

ExpdNode = namedtuple('ExpdNode', 'value forks')
ExpdNode.__repr__ \
    = lambda s: tuple.__repr__((Symbol(':{}'.format(s[0])),
                                [Symbol(n) for n in s.forks]))

BOTTOM = ExpdNode(ACCEPT, [])


class Grammar(object):

    """A grammar object containing lexical rules, syntactic rules and
    associating rule semantics.


    Missions of this class:

    * Auto-augmentation: add a singleton top rule if not explicitly
      given.

    * Auto-check:

        * Reference to undeclared symbols;
        * Unused tokens;
        * Unreachable rules;
        * LOOPs;

    """

    def __init__(self, lexes, lexpats, rules, semans=None, prece=None, lexhdls=None):
        """
        lex     : A list of lexical element names
        lexpats : A list of lexical patterns
        rules   : A list of grammar rules.
        semans  : A list of rule semantics
        prece   : A dict for operator's precedence ordinals.
        lexhdls : A dict for lexical handlers

        """
        if len(lexes) < 1:
            raise GrammarError('No lexical rules defined.')
        if len(rules) < 1:
            raise GrammarError('No grammar rules defined.')

        # Always match IGNORED patterns when all normal pattern fail.
        if IGNORED not in lexes:
            lexes.append(IGNORED)
            lexpats.append(IGNORED_PATTERN_DEFAULT)

        # Always match ERROR patterns at last.
        if ERROR not in lexes:
            lexes.append(ERROR)
            lexpats.append(ERROR_PATTERN_DEFAULT)

        # Operator precedence table may be useful.
        self.precedence = dict(prece) if prece else {}
        self.lex_handlers = dict(lexhdls) if lexhdls else {}

        # Cache terminals
        self.terminals = terminals = set()
        self.lex2pats = []
        for tmn, pat in zip(lexes, lexpats):
            # Name trailing with integer subscript indicating
            # the precedence.
            terminals.add(tmn)
            self.lex2pats.append((tmn, pat))

        # Cache nonterminals
        self.nonterminals = nonterminals = []
        self.rules = rules
        if semans:
            self.semans = semans
        else:
            self.semans = semans = [None for _ in rules]
        for rule in rules:
            if rule.lhs not in nonterminals:
                nonterminals.append(rule.lhs)

        # Sort rules with consecutive grouping of LHS.
        # rules.sort(key=lambda r: self.nonterminals.index(r.lhs))

        # This block checks completion of given grammar.
        unused_t = set(terminals).difference([IGNORED, END, ERROR])
        unused_nt = set(nonterminals).difference([nonterminals[0]])
        # Raise grammar error in case any symbol is undeclared

        for r, rl in enumerate(rules):
            for j, X in enumerate(rl.rhs):
                unused_t.discard(X)
                unused_nt.discard(X)
                if X not in terminals and X not in nonterminals:
                    msg = '\n'.join([
                        '',
                        '====================',
                        'Undeclared symbol:',
                        "* in {}`th rule {};".format(r + 1, rl),
                        "* {}`th RHS symbol `{}`.".format(j + 1, X),
                        '====================',
                        '',
                    ])
                    raise GrammarError(msg)

        # Raise warnings in case any symbol is unused.
        for t in unused_t:
            # warnings.warn('Warning: referred terminal symbol {}'.format(t))
            warnings.warn('Unreferred terminal symbol {}'.format(repr(t)))
        for nt in unused_nt:
            # warnings.warn('Warning: referred nonterminal symbol {}'.format(nt))
            warnings.warn('Unreferred nonterminal symbol {}'.format(repr(nt)))

        # Generate top rule as Augmented Grammar only if
        # the singleton top rule is not explicitly given.
        fst_rl = rules[0]
        if len(fst_rl.rhs) != 1 or 1 < [rl.lhs for rl in rules].count(fst_rl.lhs):
            tp_lhs = "{}^".format(fst_rl.lhs)
            tp_rl = Rule(tp_lhs, (fst_rl.lhs,))
            nonterminals.insert(0, tp_lhs)
            rules.insert(0, tp_rl)
            if semans is not None:
                semans.insert(0, id_func)

        self.symbols = nonterminals + [a for a in terminals if a != END]

        # Top rule of Augmented Grammar.
        self.top_rule = rules[0]
        self.top_symbol = nonterminals[0]

        # Helper for fast accessing with Trie like structure.
        self._ngraph = defaultdict(list)
        for r, rl in enumerate(self.rules):
            self._ngraph[rl.lhs].append((r, rl))

        # Prepare useful information for parsing.
        self._calc_nullable()
        self._calc_pred_trees()
        self._calc_first()

    def __repr__(self):
        return 'Grammar{{\n{}\n{}\n}}'.format(
            pp.pformat(self.lex2pats),
            pp.pformat(self.rules))

    def __getitem__(self, X):
        """Note retrieved rules are enumerated with original indices."""
        if X in self._ngraph:
            return self._ngraph[X]
        else:
            raise ValueError('No such LHS {} in grammar.'.format(repr(X)))

    def item(G, r, pos):
        """Make a pair of integers indexing the rule and active position."""
        return Item(G.rules, r, pos)

    def pred_tree(G, ntml, bottom=None):
        """Build a prediction tree for the given `ntml` by pushing rule
        symbols as nodes into a GSS structure where alternative rules
        comprise branching and left-recursion comprise cycles.  All
        the leaf nodes form the Top List representing this tree.

        A prediction tree contains 3 types of nodes:

            - Prediction node
            - Reduction node
            - Expansion node (Symbol-completion node)

        The FIRST (which are the leaf nodes) and NULLABLE (which is
        the reachability from leaves to root without bypassing
        prediction nodes).

        This tree may contain pure REDUCTION/EXPANSION-cycles due to
        LOOP in the grammar.

        For example, given the grammar

        S = A; A = B c; B = S c d

        we construct a prediction tree as follows, where (!X=Y)
        denotes reduction and [X] denotes expansion:

        S >> $

        A >> (!S=A) >> [S] >> $

        B >> c >> (!A=Bc) >> [A] >> (!S=A) >> [S] >> $

        S >> c >> A >> (!B=ScA) >> [B] >> c >> (!A=Bc) >> [A] >> (!S=A) >> [S] >> $

        But S already gets expanded, just link it back.

             c >> A >> (!B=ScA) >> [B] >> c >> (!A=Bc) >> [A] >> (!S=A) >> [S] >> $
             |                                                              |
             -----------------------------<<---------------------------------

        It is funny that the 'prediction graph' is indeed a
        'completion graph' denoting paths for completion.

        """

        global PREDICT, REDUCE
        assert ntml in G.nonterminals

        if bottom is None:
            bottom = ExpdNode(ACCEPT, [])

        # None for root/bottom.
        # - Should bottom be some "gate" to allow transplant?
        toplst = [Node((PREDICT, ntml), bottom)]
        expdd = {}

        # Each entry in top list is either a PREDICT/REDUCE node.
        z = 0
        while z < len(toplst):
            n = toplst[z]
            (act, arg), nxt = n
            #
            if act is PREDICT:
                X = arg
                if X in G.nonterminals:
                    # Discard predictive nonterminal node.
                    toplst.pop(z)
                    # Link the expansion node back.
                    if X in expdd:
                        assert isinstance(expdd[X], ExpdNode)
                        expdd[X].forks.append(nxt)
                    else:
                        # New expanded node - shared by alternative rules.
                        enxt = expdd[X] = ExpdNode(X, [nxt])
                        for r, rl in G[X]:
                            # A reduction node gating the expansion
                            # node.
                            nxt1 = Node((REDUCE, rl), enxt)
                            # Note if `rl` is a null-rule, the
                            # expansion node immediately follows the
                            # reduction node.
                            for Y in reversed(rl.rhs):
                                # New normal prediction node.
                                nxt1 = Node((PREDICT, Y), nxt1)
                            # All leaf nodes are either prediction
                            # node or reduction node.
                            assert nxt1.value[0] in [PREDICT, REDUCE]
                            toplst.append(nxt1)
                else:
                    # Path led by terminal stays.
                    z += 1

            # Strategy of eager construction:
            # When any reduction node appears in the top list, the
            # associated rule must be nullable and the subsequent
            # nodes, which are the forks from the following expansion
            # node, should be added to top list.
            elif act is REDUCE:
                # rl = arg
                # assert G.is_nullable_seq(rl.rhs), rl
                # assert isinstance(nxt, ExpdNode)
                # #
                # for fk in nxt.forks:
                #     # `fk` maybe the bottom.
                #     if fk is not bottom:
                #         (act1, arg), _ = fk
                #         if act1 == PREDICT:
                #             toplst.append(fk)
                #         elif act1 == REDUCE:
                #             # if fk not in toplst:
                #             toplst.append(fk)
                #         else:
                #             assert False, ('Invaid node following ExpdNode.')
                z += 1
            else:
                assert False, act

        # No ExpdNode is exposed to the top, since at least a REDUCE
        # node covers it, which is only possible for nullable rules.
        for n in toplst:
            assert isinstance(n, Node)
            (act, x), nxt = n
            assert \
                (act is PREDICT and x in G.terminals) or \
                (act is REDUCE and x in G.rules), (act, x)

        return toplst

    def _calc_nullable(G):

        # There may be multiple alternative nullable rules
        # for the same nonterminal, use a ddict<list> to
        # remenber them all.
        G.NULLABLE = defaultdict(list)
        candis = []

        # Find order-1 nullable.
        for rl in G.rules:
            lhs, rhs = rl
            if not rhs:
                G.NULLABLE[lhs].append(rl)
            elif all(X in G.nonterminals for X in rhs):
                candis.append(rl)

        # Find order-n nullable use stupid but clear More-Pass.
        while 1:
            has_new = False
            for rl in candis:
                lhs, rhs = rl
                if all(X in G.NULLABLE for X in rhs):
                    if rl not in G.NULLABLE[lhs]:
                        G.NULLABLE[lhs].append(rl)
                        has_new = True
                        break
            if not has_new:
                break

        # From here all NULLABLE are available.

        # Find hidden left recursions.
        G.NULL_LEAD = []
        for rl in G.rules:
            lhs, rhs = rl
            if rhs and rhs[0] in G.NULLABLE:
                G.NULL_LEAD.append(rl)

        # Compute singleton transitive derivations.
        G.DERIV1 = defaultdict(set)
        for lhs, rhs in G.rules:
            for i, X in enumerate(rhs):
                if X in G.nonterminals:
                    if all(Y in G.NULLABLE for Y in rhs[:i] + rhs[i+1:]):
                        G.DERIV1[lhs].add(X)

        # Compute LOOP clusters with representatives.
        G.LOOP = {}
        loops = []
        for ntml in G.nonterminals:
            for loop in G.find_loops(ntml):
                loops.append(loop)
                for x in loop:
                    if x not in G.LOOP:
                        G.LOOP[x] = ntml

        if loops:
            msg = '\n'.join([
                '',
                '==================================================',
                repr(G),
                ' contains loops ',
                pp.pformat(loops),
                '==================================================',
                '',
            ])
            warnings.warn(msg)

    def find_loops(G, ntml):
        """LOOP is the case that for some terminal S there is

        S => ... => S

        (not including partial LOOP like S => ... => S a)

        - Along the path the rule must be ALL nonterminals;

        - If S => .. => A and some non-singleton rule exits like
          (A=BCD), then LOOP may exits only when all except one of
          {B, C, D} are nullable.

        - Thus LOOP can be found by testing reachability through
          singleton-derivations.

        """

        paths = [[ntml]]
        while paths:
            path = paths.pop()
            X = path[-1]
            # Whole cycle from start.
            if X == path[0] and len(path) > 1:
                # cluster.update(path)
                yield path
            # Parital cycle linking to middle.
            elif X in path[:-1]:
                # j = path.index(X)
                # yield path[j:]
                pass
            # Still no cycle, try explore further.
            else:
                for nxt in G.DERIV1[X]:
                    paths.append(path + [nxt])

    def _calc_pred_trees(G):
        G.FIRST0 = {}
        for nt in reversed(G.nonterminals):
            tl = G.pred_tree(nt)
            G.FIRST0[nt] = f = set()
            # for nd in tl:
            z = 0
            while z < len(tl):
                nd = tl[z]
                if isinstance(nd, Node):
                    act, x = nd.value
                    if act is PREDICT:
                        assert x in G.terminals, x
                        f.add(x)
                    elif act is REDUCE:
                        assert x in G.rules
                        f.add(EPSILON)
                    else:
                        raise ValueError('Not valid action in toplist.')
                else:
                    assert False, nd
                z += 1

    def first_of_seq(G, seq, tail=DUMMY):
        """Find the FIRST set of a sequence of symbols. This relies on the
        precompulated indirect nullables.

        :seq:   A list of strings

        """
        assert not isinstance(seq, str)
        fs = set()
        for X in seq:
            if X in G.nonterminals:
                fs.update(G.FIRST[X])
            else:
                fs.add(X)
            if X not in G.NULLABLE:
                fs.discard(EPSILON)
                return fs
        fs.discard(EPSILON)
        # Note :tail: can also be EPSILON
        fs.add(tail)
        return fs

    def _calc_first(G):
        """Construct canonical FIRST set by iteratively augmenting FIRST0[X]
        with first_of_seq(x) for alternatives (X = x).

        """
        G.FIRST = {a: set(f) for a, f in G.FIRST0.items()}
        while 1:
            has_new = False
            for lhs, rhs in G.rules:
                f_l = G.FIRST[lhs]
                f_r = G.first_of_seq(rhs, EPSILON)
                for a in f_r:
                    if a not in f_l:
                        f_l.add(a)
                        has_new = True
            if not has_new:
                break

    def first(G, X):
        if X in G.nonterminals:
            return G.FIRST[X]
        else:
            return {X}

    def is_nullable_seq(G, seq):
        return all(X in G.NULLABLE for X in seq)

    def closure(G, I):
        """Naive CLOSURE calculation without lookaheads.

        Fig 4.32：
        SetOfItems CLOSURE(I)
            J = I.copy()
            for (A -> α.Bβ) in J
                for (B -> γ) in G:
                    if (B -> γ) not in J:
                        J.add((B -> γ))
            return J
        """

        C = I[:]
        z = 0
        while z < len(C):
            itm = C[z]
            if not itm.ended():
                if itm.actor in G.nonterminals:
                    for j, jrl in G[itm.actor]:
                        jtm = G.item(j, 0)
                        if jtm not in C:
                            C.append(jtm)
            z += 1
        return C

    def closure1_with_lookahead(G, item, a=DUMMY):
        """Fig 4.40 in Dragon Book.

        CLOSURE(I)
            J = I.copy()
            for (A -> α.Bβ, a) in J:
                for (B -> γ) in G:
                    for b in FIRST(βa):
                        if (B -> γ, b) not in J:
                            J.add((B -> γ, b))
            return J


        The lookahead must not be shared by any symbols within any instance of
        Grammar, a special value is used as the dummy.

        For similar implementations within lower-level language like
        C, this value can be replaced by any special number which
        would never represent a unicode character.

        """
        C = [(item, a)]
        z = 0
        while z < len(C):
            itm, a = C[z]
            if not itm.ended():
                if itm.actor in G.nonterminals:
                    for j, jrl in G[itm.actor]:
                        for b in G.first_of_seq(itm.look_over, a):
                            jtm = G.item(j, 0)
                            if (jtm, b) not in C:
                                C.append((jtm, b))
            z += 1
        return C


class Lexer(list):

    def __init__(self, lex2pats, lex_handlers=None, lex_handler_sources=False):
        # Raw info
        self.extend(lex2pats)
        # Compiled to re object
        self.lex2rgxs = [
            (lex, re.compile(pat) if pat else None)
            for lex, pat in lex2pats
        ]
        # Handlers
        if lex_handlers:
            self.lex_handlers = lex_handlers
        else:
            self.lex_handlers = {}
        # Source for tokenizer
        if lex_handler_sources:
            self.lex_handler_sources = lex_handler_sources
        else:
            hdl_src = {}
            for lex, hdl in lex_handlers.items():
                # src = textwrap.dedent(inspect.getsource(hdl))
                # src = src[src.index('def'):]
                # hdl_src[lex] = src
                hdl_src[lex] = marshal.dumps(hdl.__code__)
            self.lex_handler_sources = hdl_src

    @staticmethod
    def from_grammar(grammar):
        return Lexer(grammar.lex2pats, grammar.lex_handlers)

    def tokenize(self, inp, with_end):
        """Perform lexical analysis with given input string and yield matched
        tokens with defined lexical patterns.

        * Ambiguity is resolved by the definition order.

        It must be reported if `pos` is not strictly increasing when
        any of the valid lexical pattern matches zero-length string!!
        This might lead to non-termination.

        Here only matches with positive length is retrieved, though
        expresiveness may be harmed by this restriction.

        """
        lex2rgxs = self.lex2rgxs
        lex_handlers = self.lex_handlers

        pos = 0
        while pos < len(inp):
            # raw string match
            raw_match = False
            # re match
            n = None
            m = None
            for cat, rgx in lex2rgxs:
                # raw
                if rgx is None:
                    if inp.startswith(cat, pos):
                        yield Token(pos, cat, cat, cat)
                        pos += len(cat)
                        raw_match = True
                        break
                # re
                else:
                    m = rgx.match(inp, pos=pos)
                    # The first match with non-zero length is yielded.
                    if m and len(m.group()) > 0:
                        n = cat
                        break
            if raw_match:
                continue
            elif m:
                assert isinstance(n, str)
                if n == IGNORED:
                    # Need IGNORED handler?
                    at, pos = m.span()
                elif n == ERROR:
                    # Call ERROR handler!
                    at, pos = m.span()
                    lxm = m.group()
                    if ERROR in lex_handlers:
                        # Suppress error token and call handler.
                        lex_handlers[ERROR](lxm)
                        # yield Token(at, ERROR, lxm, h(lxm))
                    else:
                        # Yield error token when no handler available.
                        yield Token(at, ERROR, lxm, lxm)
                else:
                    at, pos = m.span()
                    lxm = m.group()
                    if n in lex_handlers:
                        # Call normal token handler.
                        h = lex_handlers[n]
                        # Bind semantic value.
                        yield Token(at, n, lxm, h(lxm))
                    else:
                        yield Token(at, n, lxm, lxm)
            else:
                # Report unrecognized Token here!
                msg = '\n'.join([
                    '',
                    '=========================',
                    'No defined pattern starts with char `{}` @{}'.format(inp[pos], pos),
                    '',
                    '* Consumed input: ',
                    repr(inp[:pos]),
                    '=========================',
                    '',
                ])
                raise GrammarError(msg)
        if with_end:
            yield Token(pos, END, END_PATTERN_DEFAULT, None)

    def dumps(self):
        fl = '\n'.join([
            '## This file is generated. Do not modify.',
            '',
            '## Lexer$BEGIN',
            '',
            'lex2pats = \\',
            # textwrap.indent(pp.pformat(list(self)), '    '),
            '    ' + pp.pformat(list(self)).replace('\n', '\n    '),
            '',
            'lex_handler_sources = \\',
            # textwrap.indent(pp.pformat(self.lex_handler_sources), '    '),
            '    ' + pp.pformat(self.lex_handler_sources).replace('\n', '\n    '),
            '',
            '## Lexer$END',
            '',
        ])
        return fl

    @classmethod
    def loads(cls, src, glb):
        ctx = {}
        ctx1 = {}
        exec(src, glb, ctx)
        lex_handlers = {}
        for k, lxsrc in ctx['lex_handler_sources'].items():
            # exec(lxsrc, glb, ctx1)
            # lex_handlers[k] = ctx1.pop(k)
            co = marshal.loads(lxsrc)
            hdl = types.FunctionType(co, glb)
            lex_handlers[k] = hdl
        ctx['lex_handlers'] = lex_handlers
        return Lexer(**ctx)


class assoclist(list):

    """This class intends to be cooperated with metaclass definition
    through __prepare__ method. When returned in __prepare__, it will
    be used by registering class-level method definitions. Since it
    overrides the setter and getter of default `dict` supporting class
    definition, it allows repeated method declaration to be registered
    sequentially in a list. As the result of such class definition,
    declarated stuff can be then extracted in the __new__ method in
    metaclass definition.

    Assoclist :: [(<key>, <value>)]

    """

    def __setitem__(self, k, v):
        """Overides alist[k] = v"""
        ls = super(assoclist, self)
        ls.append((k, v))

    def __getitem__(self, k0):
        """Overrides: ys = alist[k]"""
        vs = [v for k, v in self if k == k0]
        if vs:
            return vs[0]
        else:
            raise KeyError('No such attribute.')

    def items(self):
        """Yield key-value pairs."""
        for k, v in self:
            yield (k, v)


class cfg(type):

    """This metaclass performs accumulation for all declared grammar
    lexical and syntactical rules at class level, where declared
    variables are registered as lexical elements and methods are
    registered as Rule objects defined above.

    """

    @staticmethod
    def read_from_raw_lists(*lsts):
        """Extract named objects from a sequence of lists and translate some
        of them into something necessary for building a grammar
        object.

        Note the ORDER of registering lexical rules matters. Here it is assumed
        the k-v pairs for such rules preserve orignial declaration order.

        """

        docstr = ''
        lexes = []
        lexpats = []
        lexhdls = {}            # Handler for some specified lexical pattern.
        rules = []
        semans = []
        prece = {}
        attrs = []

        for lst in lsts:
            for k, v in lst:

                # Docstring for grammar
                if k == '__doc__':
                    docstr = v

                # Built-ins are of no use
                elif k.startswith('__') and k.endswith('__'):
                    continue

                # Handle methods.
                elif callable(v):

                    # For lexical pattern rule handlers
                    # - Associated handler for declared pattern
                    if k == ERROR:
                        lexhdls[ERROR] = v
                    elif k in lexes:
                        par = v.__code__.co_varnames[:v.__code__.co_argcount]
                        assert len(par) == 1, \
                            "Lexical handler must have only one parameter.".format(par)
                        assert k not in lexhdls, \
                            "Repeated handler declaration for {}!".format(k)
                        lexhdls[k] = v

                    # - Pattern and handler together
                    # # - This scheme may be too much tweaked.
                    # elif hasattr(v, '__annotations__') and 'lex' in v.__annotations__:
                    #     ann = v.__annotations__
                    #     assert v.__code__.co_argcount == 1, \
                    #         'Single argument handler required for {}.'.format(repr(k))
                    #     assert k not in lexhdls, \
                    #         'Repeated handler declaration for {}!'.format(repr(k))
                    #     lexes.append(k)
                    #     lexpats.append(ann['lex'])
                    #     lexhdls[k] = v
                    #     if 'return' in ann:
                    #         prece[k] = ann['return']

                    # For rule declarations
                    else:
                        rule = Rule.read(v)
                        if rule in rules:
                            raise GrammarError('{}`th rule {}: Repeated declaration!\n'.format(
                                len(rules), rule))
                        rules.append(rule)
                        semans.append(v)

                # Lexical declaration allows repeatition!!
                # - String pattern without precedence
                elif isinstance(v, str):
                    lexes.append(k)
                    lexpats.append(v)

                # - String pattern with precedence
                elif isinstance(v, tuple):
                    assert len(v) == 2, 'Lexical pattern as tuple should be of length 2.'
                    pat, prc = v
                    assert isinstance(pat, str) and isinstance(prc, int), \
                        'Lexical pattern as tuple should be of type (str, int)'
                    assert k not in [IGNORED, ERROR, END], \
                        'Special token {} allow no precedence.'.format(repr(k))
                    lexes.append(k)
                    lexpats.append(pat)
                    if k in prece:
                        assert prece[k] == prc, \
                            ('Specifying conflicting precedence for '
                             '{}: {} and {}.'
                             .format(repr(k), prece[k], prc))
                    prece[k] = prc

                # Handle normal private attributes/methods.
                else:
                    attrs.append((k, v))

        # Precedence values should be restricted by dedicated operators.

        # Default matching order of special patterns:

        g = Grammar(lexes, lexpats, rules, semans, prece, lexhdls)
        if docstr:
            g.__doc__ = docstr

        return g

    @classmethod
    def __prepare__(mcls, n, bs, **kw):
        "Prepare a collector list."
        return assoclist()

    def __new__(mcls, n, bs, accu):
        """Read a grammar from the collector list `accu`. """

        return cfg.read_from_raw_lists(accu)

    @staticmethod
    def extract_list(decl):
        """Retrieve structural information of the function `decl` (i.e. the
        body and its components IN ORDER). Then transform these
        information into a list of lexical elements and rules as well
        as semantics.

        """
        import textwrap

        # The context for the rule semantics is `decl`'s belonging
        # namespace, here `__globals__`.
        glb_ctx = decl.__globals__

        # The local context for registering a function definition
        # by `exec`.
        lcl_ctx = {}

        # Prepare the source. If the `decl` is a method, the source
        # then contains indentation spaces thus should be dedented in
        # order to be parsed independently.  `textwrap.dedent`
        # performs dedenation until there are no leading spaces.
        src = inspect.getsource(decl)
        src = textwrap.dedent(src)

        # Parse the source to Python syntax tree.
        t = ast.parse(src)

        lst = []

        for obj in t.body[0].body:

            if isinstance(obj, ast.Assign):
                if type(obj.value) in [ast.Str, ast.Tuple]:
                    # Temporary context.
                    ctx1 = OrderedDict()
                    md = ast.Module([obj])
                    co = compile(md, '<ast>', 'exec')
                    # Execute the assignments with given context!
                    exec(co, glb_ctx, ctx1)
                    for kv in ctx1.items():
                        lst.append(kv)

            elif isinstance(obj, ast.FunctionDef):
                name = obj.name
                # Conventional fix
                ast.fix_missing_locations(obj)
                # `ast.Module` is the unit of program codes.
                md = ast.Module(body=[obj])
                # Compile a module into ast with '<ast>' mode
                # targeting 'exec'.
                code = compile(md, '<ast>', 'exec')
                # Register function into local context, within the
                # circumference of global context.
                exec(code, glb_ctx, lcl_ctx)
                func = lcl_ctx.pop(name)
                lst.append((name, func))

            else:
                # Ignore structures other than `Assign` and `FuncionDef`
                pass

        return lst

    @staticmethod
    def v2(func):
        lst = cfg.extract_list(func)
        return cfg.read_from_raw_lists(lst)


grammar = cfg.v2


class VerboseReader(object):

    class LexLogger(object):

        def __init__(self):
            self.lexes = []
            self.lexpats = []
            self.lexhdls = {}
            self.prece = {}

        def __call__(self, p=None, rgx=True, **kw):
            name, pat = kw.popitem()
            assert not kw
            if rgx:
                self.lexes.append(name)
                self.lexpats.append(pat)
            # else:
            #     self.lexes.append()
            if p is not None:
                assert isinstance(p, int)
                self.prece[name] = p
            return lambda func: self.lexhdls.__setitem__(name, func)


    class RuleLogger(list):

        def __init__(self):
            self.rules = []
            self.semans = []

        def __call__(self, seman):
            self.rules.append(Rule.read(seman))
            self.semans.append(seman)


def verbose(func):
    """Read a grammar declared with verbosity. This is a more
    straightforward but less concise usage.

    :func: must accept exactly 2 parameters, i.e. a lex-logger and a
    rule-logger.

    """
    ll = VerboseReader.LexLogger()
    rl = VerboseReader.RuleLogger()
    func(ll, rl)
    args = {}
    args.update(ll.__dict__)
    args.update(rl.__dict__)
    g = Grammar(**args)
    return g



"""The above parts are utitlies for grammar definition and extraction
methods of grammar information.

The following parts supplies utilities for parsing.

"""

# The object ParseLeaf is semantically indentical to the object Token.
ParseLeaf = Token


def meta(cls_parser):
    """This decorator hangs a static attribute 'meta' to the decorated
    parser class, which is itself a nested class for specifying a
    grammar+parser declaration directly, such as:

    class MyGrammarWithLALRParser(metaclass=LALR.meta):
        # lexical rules
        # syntactic rules with semantics

    """

    class meta(cfg):
        def __new__(mcls, n, bs, kw):
            grammar = cfg.__new__(mcls, n, bs, kw)
            return cls_parser(grammar)

    setattr(cls_parser, 'meta', meta)

    return cls_parser


class ParseTree(object):
    """ParseTree is the basic object representing a valid parsing.

    Note the node value is a :Rule: object rather than a grammar
    symbol, which allows the tree to be traversed for translation with
    the rule's semantics.

    """

    def __init__(self, rule, subs, seman):
        # Contracts
        assert isinstance(rule, Rule)
        for sub in subs:
            assert isinstance(sub, ParseTree) or isinstance(sub, ParseLeaf)
        self.rule = rule
        self.subs = subs
        self.seman = seman

    def translate(tree, trans_tree=None, trans_leaf=None, as_tuple=False):
        """Postorder tree traversal with double-stack scheme.
        """
        # Direct translation
        # Aliasing push/pop, simulating stack behavior
        push, pop = list.append, list.pop
        sstk = [(PREDICT, tree)]
        astk = []
        while sstk:
            act, t = pop(sstk)
            if act == REDUCE:
                args = []
                for _ in t.subs:
                    a = pop(astk)
                    args.append(a)
                args = args[::-1]
                # Apply semantics
                if as_tuple:
                    push(astk, (Symbol(t.rule.lhs), args))
                elif trans_tree:
                    push(astk, trans_tree(t, args))
                else:
                    push(astk, t.seman(*args))
            elif isinstance(t, ParseLeaf):
                # Apply semantics
                if as_tuple:
                    push(astk, t)
                elif trans_leaf:
                    push(astk, trans_leaf(t))
                else:
                    push(astk, t.value)
            elif isinstance(t, ParseTree):
                # mark reduction
                push(sstk, (REDUCE, t))
                for sub in reversed(t.subs):
                    push(sstk, (PREDICT, sub))
            else:
                assert False, (act, t)
        assert len(astk) == 1
        return astk.pop()

    def to_tuple(self):
        return self.translate(as_tuple=True)

    def __eq__(self, other):
        if isinstance(other, ParseTree):
            return self.to_tuple() == other.to_tuple()
        elif isinstance(other, Iterable):
            return self.to_tuple() == other
        else:
            return False

    def __repr__(self):
        """Use pformat to handle structural hierarchy."""
        return pp.pformat(self.to_tuple())


class ParserBase(object):

    """Abstract class for both deterministic/non-deterministic parsers.
    They both have methods :parse_many: and :interpret_many:, while
    non-deterministic parsers retrieve singleton parse list with these
    methods.

    """

    def __init__(self, grammar):
        self.lexer = Lexer.from_grammar(grammar)
        self.semans = grammar.semans

    def __repr__(self):
        # Polymorphic representation without overriding
        # raise NotImplementedError('Parser should override __repr__ method. ')
        return self.__class__.__name__ + '-Parser-{}'.format(self.grammar)

    def parse_many(self, inp, interp=False):
        # Must be overriden
        raise NotImplementedError('Any parser should have a parse method.')

    def interpret_many(self, inp):
        # if not self.semans:
        #     raise ParserError('Semantics not specified. Only `parse` is supported.')
        # else:
        return self.parse_many(inp, interp=True)

    def parse(self, inp, interp=False):
        raise NotImplementedError('For non-deterministic parsers use `parse_many` instead.')

    def interpret(self, inp):
        raise NotImplementedError('For non-deterministic parsers use `interpret_many` instead.')


class ParserDeterm(ParserBase):

    """Abstract class for deterministic parsers. While the *parse_many* method
    tends to return a list of results and deterministic parsers yield
    at most ONE result, *parse* and *interpret* return this result
    directly.

    """

    def parse_many(self, inp, interp=False):
        return [self.parse(inp, interp=interp)]

    def interpret_many(self, inp):
        return [self.interpret(inp)]

    def parse(self, inp, interp):
        raise NotImplementedError('Deterministic parser should have `parse` method.')

    def interpret(self, inp):
        if not self.semans:
            raise ParserError('Semantics not specified. Only `parse` is supported.')
        else:
            return self.parse(inp, interp=True)


@meta
class Earley(ParserBase):

    """Earley parser is able to recognize ANY Context-Free Language
    properly using the technique of dynamic programming. It performs
    non-deterministic parsing since all parse results due to potential
    ambiguity of given grammar should be found.

    WARNING: In this implementation, grammars having NULLABLE-LOOPs
    like:

      A -> A B
      A ->
      B -> b
      B ->

    where A => ... => A consuming no input tokens, may suffer from
    nontermination under on-the-fly computation of parse
    trees. However, recognition without building parse trees stays
    intact.

    """

    def __init__(self, grammar):
        super(Earley, self).__init__(grammar)
        self.grammar = grammar

    def recognize(self, inp):
        """Naive Earley's recognization algorithm. No parse produced.

        """
        G = self.grammar
        L = self.lexer
        S = [[(0, G.item(0, 0))]]

        for k, tok in enumerate(L.tokenize(inp, with_end=True)):

            C = S[-1]
            C1 = []

            z_j = 0
            while z_j < len(C):
                j, jtm = C[z_j]
                if not jtm.ended():
                    # Prediction/Scanning is not needed when
                    # recognizing END-Token.
                    if not tok.is_END():
                        # Prediction: find nonkernels
                        if jtm.actor in G.nonterminals:
                            for r, rule in G[jtm.actor]:
                                ktm = G.item(r, 0)
                                new = (k, ktm)
                                if new not in C:
                                    C.append(new)
                                if not rule.rhs:
                                    # Directly NULLABLE
                                    enew = (j, jtm.shifted)
                                    if enew not in C:
                                        C.append(enew)
                        # Scanning
                        elif jtm.actor == tok.symbol:
                            C1.append((j, jtm.shifted))
                # Completion
                else:
                    for (i, itm) in S[j]:
                        if not itm.ended() and itm.actor == jtm.target:
                            new = (i, itm.shifted)
                            if new not in C:
                                C.append(new)
                z_j += 1
            if not C1:
                if not tok.is_END():
                    msg = '\n'.join([
                        '',
                        '=========================',
                        'Unrecognized {}'.format(tok),
                        'Choked active ItemSet: ',
                        pp.pformat(C),
                        '',
                        'States:',
                        '',
                        pp.pformat(S),
                        '',
                        '=========================',
                    ])
                    raise ParserError(msg)
            else:
                # Commit proceeded items (by Scanning) as item set.
                S.append(C1)

        return S

    def parse_chart(self, inp):
        """Perform general chart-parsing framework method with Earley's
        recognition algorithm. The result is a chart, which
        semantically includes all possible parsing results.

        """

        G = self.grammar

        # `chart[i][j]` registers active items which are initialized
        # @i and still alive @j.
        chart = self.chart = defaultdict(lambda: defaultdict(set))

        agenda = [(0, G.item(0, 0))]

        for k, tok in enumerate(G.tokenize(inp, True)):
            agenda1 = []
            while agenda:
                j, jtm = agenda.pop()
                # Directly register intermediate state item.
                if jtm not in chart[j][k]:
                    chart[j][k].add(jtm)
                if not jtm.ended():
                    if not tok.is_END():
                        # Prediction
                        if jtm.actor in G.nonterminals:
                            for r, rule in G[jtm.actor]:
                                ktm = G.item(r, 0)
                                if ktm not in chart[k][k]:
                                    chart[k][k].add(ktm)
                                    agenda.append((k, ktm))
                                # Prediction over NULLABLE1 symbol
                                if not rule.rhs:
                                    jtm1 = jtm.shifted
                                    if jtm1 not in chart[k][k]:
                                        chart[j][k].add(jtm1)
                                        agenda.append((k, jtm1))
                        # Scanning
                        elif jtm.actor == tok.symbol:
                            chart[k][k+1].add(jtm.shifted)
                            chart[j][k+1].add(jtm.shifted)
                            agenda1.append((j, jtm.shifted))
                else:
                    # Completion
                    for i in range(j+1):
                        if j in chart[i]:
                            for itm in chart[i][j]:
                                if not itm.ended() and itm.actor == jtm.target:
                                    if itm.shifted not in chart[i][k]:
                                        agenda.append((i, itm.shifted))
            if agenda1:
                agenda = agenda1
            # else:
            #     raise ParserError('Fail: empty new agenda by {}\nchart:\n{}'.format(tok,
            #         pp.pformat(chart)))

        return chart

    def parse_forest(self, inp):
        """Construct single-threaded parse trees (the Parse Forest based upon
        the underlying Graph Structured Stack and from another
        perspective, the chart) during computing Earley item sets.

        The most significant augmentation w.R.t. the naive recognition
        algorithm is that, when more than one completed items @jtm in
        the agenda matches one parent @(itm)'s need, the parent items'
        stacks need to be forked/copied to accept these different @jtm
        subtrees as a feasible shift.

        Since parse trees are computed on-the-fly, the result is a set
        of feasible parse trees.

        """
        G = self.grammar
        L = Lexer.from_grammar(G)
        # state :: {<Item>: [<Stack>]}
        s0 = {(0, G.item(0, 0)): [()]}
        ss = self.forest = []
        ss.append(s0)

        # Tokenizer with END token to force the tailing completion
        # pass.
        for k, tok in enumerate(L.tokenize(inp, with_end=True)):
            # Components for the behavior of One-Pass transitive closure
            #
            # - The accumulated set of items/stacks as final closure.
            #   It plays the role of the chart column k
            s_acc = ss[-1]
            # - The set of active items to be processed, i.e. AGENDA
            s_act = dict(s_acc)
            # - The initial set for after reading current token
            s_acc1 = {}
            while 1:
                # - The items induced while processing items in the
                #   AGENDA, named AUGMENTATION AGENDA.
                s_aug = {}
                for (j, jtm), j_stks in s_act.items():

                    # PREDICTION
                    if not jtm.ended():

                        # *Problem of nullability* dealt with
                        # enhanced-predictor. See book chapter
                        # #Parsing Techniques# 7.2.3.2 and references.
                        #
                        # In general cases, any INDIRECTLY NULLABLE
                        # symbol E with rule (E->αBC) referring (B->)
                        # and (C->) would finally be reduced, since
                        # [j,(E->α.BC)] ensures [j,(E->αB.C)] to be
                        # appended and waiting, which when processed
                        # ensures [j,(E->αBC.)] to be waiting further,
                        # causing [j,(E->αBC.)] finally to be
                        # completed. Thus ONE-PASS can escape no
                        # NULLABLEs.
                        #
                        # INVARIANT during processing:
                        #
                        # - No completion can induce any new
                        #   prediction at this position k (Thus no
                        #   prediction can induce any new completion).
                        #
                        # - Each completion induces shfited items in
                        #   the next item set independently
                        #
                        # Note only the DIRECT NULLABLILIDY is
                        # concerned, rather than INDIRECT. Since the
                        # computation of item set implictly covers
                        # the computation of INDIRECT NULLABILITY
                        # already.
                        #
                        # More details see the referenced book.
                        # FIXME: Seems rigorous?
                        # if not tok.is_END():
                        # Even when the token has ended must the predictor
                        # find some potentially ended items after looking
                        # over nullable symbols!

                        if jtm.actor in G.nonterminals:
                            for r, rule in G[jtm.actor]:
                                new = (k, G.item(r, 0))
                                if new not in s_acc:
                                    if rule.rhs:
                                        s_aug[new] = [()]
                                    else:
                                        # Nullable completion and
                                        # prediction.
                                        new0 = (j, jtm.shifted)
                                        j_tr = ParseTree(rule, [], G.semans[r])
                                        for j_stk in j_stks:
                                            if new0 not in s_aug:
                                                s_aug[new0] = [j_stk + (j_tr,)]
                                            else:
                                                s_aug[new0].append(j_stk + (j_tr,))
                        # SCANNING
                        elif jtm.actor == tok.symbol:
                            if (j, jtm.shifted) not in s_acc1:
                                s_acc1[j, jtm.shifted] = []
                            for j_stk in j_stks:
                                s_acc1[j, jtm.shifted].append(j_stk + (tok,))
                    # COMPLETION
                    else:
                        for j_stk in j_stks:
                            j_tr = ParseTree(jtm.rule, j_stk, G.semans[jtm.r])
                            for (i, itm), i_stks in ss[j].items():
                                if not itm.ended() and itm.actor == jtm.target:
                                    new = (i, itm.shifted)
                                    if new not in s_aug:
                                        s_aug[new] = []
                                    for i_stk in i_stks:
                                        s_aug[new].append(i_stk + (j_tr,))
                if s_aug:
                    # Register new AGENDA items as well as their
                    # corresponding completed stacks
                    for (i, itm), i_stks in s_aug.items():
                        if (i, itm) in s_acc:
                            s_acc[i, itm].extend(i_stks)
                        else:
                            s_acc[i, itm] = i_stks
                    # Next pass of AGENDA
                    s_act = s_aug
                else:
                    # No new AGENDA items
                    break

            if not tok.is_END():
                ss.append(s_acc1)

        return ss

    def parse_many(self, inp, interp=False):
        """Fully parse. Use parse_forest as default parsing method and final
        parse forest constructed is returned.

        """
        G = self.grammar
        forest = self.parse_forest(inp)
        fin = []
        for (i, itm), i_stks in forest[-1].items():
            if i == 0 and itm.r == 0 and itm.ended():
                for i_stk in i_stks:
                    assert len(i_stk) == 1, 'Top rule should be Singleton rule.'
                    # tree = ParseTree(G.top_rule, i_stk)
                    tree = i_stk[0]
                    if interp:
                        fin.append(tree.translate())
                    else:
                        fin.append(tree)
        return fin


class LR(ParserBase):
    """Abstract class for LR parsers supporting dumps/loads methods."""

    def __init__(self, grammar=None, dict=None):
        if grammar is not None:
            super(LR, self).__init__(grammar)
            self.lexer = Lexer.from_grammar(grammar)
            self._build_automaton(grammar)
            try:
                seman_sources = []
                for f in self.semans:
                    if f:
                        # src = textwrap.dedent(inspect.getsource(f))
                        # src = src[src.index('def'):]
                        # seman_sources.append(src)
                        seman_sources.append(marshal.dumps(f.__code__))
                    else:
                        seman_sources.append(None)
                self.seman_sources = seman_sources
            except OSError:
                pass
        elif dict is not None:
            for k, v in dict.items():
                setattr(self, k, v)

    def __repr__(self):
        return '{}-Parser-for-Grammar\n{}'.format(
            self.__class__.__name__,
            pp.pformat(self.rules))

    def _build_automaton(self, G):
        """Initiate attributes and build automaton."""
        raise NotImplementedError()

    def dumps(self):
        """Dump the parser automaton into a Python file string."""
        fl = self.lexer.dumps()
        fl += '\n\n## Parser$BEGIN\n'

        for k in 'precedence rules seman_sources Ks ACTION GOTO'.split():
            if k == 'rules':
                v = pp.pformat(
                    [tuple(rl) for rl in self.rules])
            elif k == 'Ks':
                v = pp.pformat(
                    [[tm.index_pair for tm in K]
                     for K in self.Ks])
            else:
                v = pp.pformat(
                    getattr(self, k))
            fl += '\n'.join([
                '',
                '{} = \\'.format(k),
                '    ' + v.replace('\n', '\n    '),
                '',
            ])
        fl += '\n## Parser$END\n'
        return fl

    @classmethod
    def loads(cls, s, glb):
        """Specify global context `glb` explicitly for the lexer and parser to
        be read (normally `globals()`) if the semantics are not pure
        (rely on global variables).

        """
        ctx = {}
        ctx1 = {}
        exec(s, glb, ctx)

        # Read lexer
        lex2pats = ctx.pop('lex2pats')
        lxsrcs = ctx.pop('lex_handler_sources')
        lex_handlers = {}
        for k, src in lxsrcs.items():
            # exec(src, glb, ctx1)
            # lex_handlers[k] = ctx1.pop(k)
            co = marshal.loads(src)
            hdl = types.FunctionType(co, glb)
            lex_handlers[k] = hdl
        lexer = Lexer(lex2pats, lex_handlers, lxsrcs)
        ctx['lexer'] = lexer

        # Read LR-tables
        semans = []
        for sm_src in ctx['seman_sources']:
            if sm_src:
                # exec(sm_src, glb, ctx1)
                # semans.append(ctx1.popitem()[-1])
                co = marshal.loads(sm_src)
                sm = types.FunctionType(co, glb)
                semans.append(sm)
            else:
                semans.append(None)
        ctx['semans'] = semans
        rules = []
        for lhs, rhs in ctx['rules']:
            rules.append(Rule(lhs, rhs))
        ctx['rules'] = rules
        Ks = []
        for Kp in ctx['Ks']:
            K = []
            for r, pos in Kp:
                K.append(Item(rules, r, pos))
            Ks.append(K)
        ctx['Ks'] = Ks

        # Just use the context as arguments.
        return cls(dict=ctx)

    def dump(self, filename):
        with open(filename, 'w') as f:
            f.write(self.dumps())

    @classmethod
    def load(cls, filename, glb):
        with open(filename, 'r') as f:
            return cls.loads(f.read(), glb)


# TODO:
#
# LR parsers can indeed corporate pre-order semantics. Each ItemSet
# has prepared a set of prediction paths, while each GOTO action
# confirms which path to go and triggers the corresponding semantics
# (though at most ONE token gets consumed already).
#
@meta
class GLR(LR):

    """GLR parser is a type of non-deterministic parsers which produces
    parse forest rather than exactly one parse tree. The traversal
    order of semantic behaviors of subtrees under the same parent rule
    should be preserved until full parse is generated. This means
    execution of semantic behavior during parsing process is banned.

    Here the assumption of LR(0) grammar is assumed. Nontermination of
    reduction process may happen for Non-LR(0) grammars, e.g. for the
    following grammar

      S -> A S
      S -> b
      A -> ε

    there is an ItemSet[i], which is

      {(S ->  A.S),
       (S -> .A S),
       (S -> .b),
       (A -> .)}

    and it is clear that

      - ('reduce', (A -> .)) in ACTION[i]
      - GOTO[i][A] == i

    thus during such reduction the stack [... i] keeps growing into
    [... i i i] nonterminally with no consumption of the next token.

    There are methods to detect such potential LOOPs, which are yet to
    be integrated.

    """

    def __init__(self, *a, **kw):
        super(GLR, self).__init__(*a, **kw)

    def _build_automaton(self, G):

        """Calculate general LR(0)-Item-Sets with no respect to look-aheads.
        Each conflict is registered into the parsing table. For
        practical purposes, these conflicts should be reported for the
        grammar writer to survey the conflicts and experiment with
        potential ambiguity, thus achieving better inspection into the
        characteristics of the grammar itself.

        For LR(0) grammars, the performance of GLR is no significantly
        worse than the LALR(1) parser.

        """

        self.rules = G.rules
        self.semans = G.semans

        self.precedence = G.precedence

        # Kernels
        self.Ks = Ks = [[G.item(0, 0)]]
        self.GOTO = GOTO = []
        self.ACTION = ACTION = []

        # Construct LR(0)-DFA
        i = 0
        while i < len(Ks):

            I = Ks[i]

            iacts = {REDUCE: [], SHIFT: {}}
            igotoset = OrderedDict()

            for itm in G.closure(I):
                if itm.ended():
                    iacts[REDUCE].append(itm.r)
                else:
                    X = itm.actor
                    jtm = itm.shifted
                    if X not in igotoset:
                        igotoset[X] = []
                    if jtm not in igotoset[X]:
                        igotoset[X].append(jtm)

            igoto = {}
            for X, J in igotoset.items():
                J.sort()
                if J not in Ks:
                    Ks.append(J)
                j = Ks.index(J)
                igoto[X] = j
                iacts[SHIFT][X] = j

            ACTION.append(iacts)
            GOTO.append(igoto)

            i += 1

    def parse_many(self, inp, interp=False):
        """Parse input with BFS scheme, where active forks all get reduced to
        synchronized states, so as to accept the active input token.

        """

        # G = self.grammar
        L = self.lexer
        GOTO = self.GOTO
        ACTION = self.ACTION
        rules = self.rules
        semans = self.semans

        # Agenda is a list of tuples. Each tuple is two parallel
        # GSS's, (gss-stack<state>, gss-stack<subtree>).
        agenda = [(Node(0, START), START)]

        results = []

        # BFS scheme with synchronization.
        for k, tok in enumerate(L.tokenize(inp, True)):

            agenda1 = []

            while agenda:

                ss, ts = agenda.pop()

                redus = ACTION[ss.value][REDUCE]
                shfts = ACTION[ss.value][SHIFT]

                # REDUCE: Note the (ended) items triggering reduction
                # action may induce further reduction items.
                for r in redus:
                    rule = rules[r]
                    seman = semans[r]
                    ss1, ts1 = ss, ts
                    subs = deque()
                    for _ in rule.rhs:
                        # Register the peeked element into subtrees.
                        subs.appendleft(ts1.value)
                        # 'Pop' from GSS.
                        ss1 = ss1.next
                        ts1 = ts1.next
                    if r == 0:
                        if tok.is_END():
                            # NOTE: Translation on-the-fly should not
                            # be supported due to possibly impure
                            # semantics.
                            results.append(subs[0]) # Unaugmented top
                    else:
                        tr = ParseTree(rule, subs, seman)
                        # 'Push/Augment' GSS.
                        ss1 = Node(GOTO[ss1.value][rule.lhs], ss1)
                        ts1 = Node(tr, ts1)
                        agenda.append((ss1, ts1))

                # SHIFT
                if not tok.is_END():
                    if tok.symbol in shfts:
                        ss = Node(GOTO[ss.value][tok.symbol], ss)
                        ts = Node(tok, ts)
                        agenda1.append((ss, ts))

            if not tok.is_END() and not agenda1:
                raise ParserError('No parse.')
            else:
                agenda = agenda1

        if interp:
            return [t.translate() for t in results]
        else:
            return results


@meta
class LALR(LR, ParserDeterm):

    """Look-Ahead Leftmost-reading Rightmost-derivation (LALR) parser may
    be the most widely used parser variant. It has almost the same
    automaton like GLR, with potential ambiguity eliminated by raising
    SHIFT/REDUCE and REDUCE/REDUCE conflicts.

    Due to its deterministic nature and table-driven process does it
    have linear-time performance.

    """

    def __init__(self, *a, **kw):
        super(LALR, self).__init__(*a, **kw)

    def _build_automaton(self, G):

        self.rules = G.rules
        self.precedence = G.precedence

        Ks = [[G.item(0, 0)]]   # Kernels
        GOTO = []

        # Calculate LR(0) item sets and GOTO
        i = 0
        while i < len(Ks):

            K = Ks[i]

            # Use OrderedDict to preserve order of found goto's.
            igotoset = OrderedDict()

            # SetOfItems algorithm
            for itm in G.closure(K):
                # If item (A -> α.Xβ) has a goto.
                if not itm.ended():
                    X, jtm = itm.actor, itm.shifted
                    if X not in igotoset:
                        igotoset[X] = []
                    if jtm not in igotoset[X]:
                        igotoset[X].append(jtm)

            # Register `igotoset` into global goto.
            igoto = {}
            for X, J in igotoset.items():
                # The Item-Sets should be treated as UNORDERED! So
                # sort J to identify the Lists with same items.
                # J = sorted(J, key=lambda i: (i.r, i.pos))
                J.sort()
                if J not in Ks:
                    Ks.append(J)
                j = Ks.index(J)
                igoto[X] = j

            GOTO.append(igoto)

            i += 1

        # The table :spont: represents the spontaneous lookaheads at
        # first. But it can be used for in-place updating of
        # propagated lookaheads. After the whole propagation process,
        # :spont: is the final lookahead table.
        spont = [OrderedDict((itm, set()) for itm in K) for K in Ks]

        # Initialize spontaneous END token for the top item set.
        init_item = Ks[0][0]
        spont[0][init_item].add(END)
        for ctm, a in G.closure1_with_lookahead(init_item, END):
            if not ctm.ended():
                X = ctm.actor
                j0 = GOTO[0][X]
                spont[j0][ctm.shifted].add(a)

        # Propagation table, registers each of the GOTO target item
        # which is to be applied with propagation from its corresponding
        # source item.
        propa = [[] for _ in Ks]

        for i, K in enumerate(Ks):
            for ktm in K:
                C = G.closure1_with_lookahead(ktm, DUMMY)
                for ctm, a in C:
                    if not ctm.ended():
                        X = ctm.actor
                        j = GOTO[i][X]
                        if a != DUMMY:
                            spont[j][ctm.shifted].add(a)
                        else:
                            # Propagation from KERNEL item `ktm` to
                            # its belonging non-kernel item `ctm`,
                            # which is shifted into `j`'th item set
                            # (by its actor). See ALGO 4.62 in Dragon
                            # book.
                            propa[i].append((ktm, j, ctm.shifted))

        table = spont

        # MORE-PASS propagation
        prop_passes = 1
        while True:
            brk = True
            for i, _ in enumerate(Ks):
                for itm, j, jtm in propa[i]:
                    lks_src = table[i][itm]
                    lks_tar = table[j][jtm]
                    for a in lks_src:
                        if a not in lks_tar:
                            lks_tar.add(a)
                            brk = False
            if brk:
                break
            else:
                prop_passes += 1

        self.Ks = Ks
        self.GOTO = GOTO

        # Now all goto and lookahead information is available.

        # Construct ACTION table
        ACTION = [{} for _ in table]

        # SHIFT for non-ended items
        # - The argument of SHIFT is the target item set index.
        # Since no SHIFT/SHIFT conflict exists, register
        # all SHIFT information firstly.
        for i, xto in enumerate(GOTO):
            for X, j in xto.items():
                if X in G.terminals:
                    ACTION[i][X] = (SHIFT, j)

        # REDUCE for ended items
        # - The argument of REDUCE is the index of the rule to be reduced.
        # SHIFT/REDUCE and REDUCE/REDUCE conflicts are
        # to be found.
 
        def intermediate_info():
            return '\n'.join([
                '- {}'.format(G),
                '',
                '- States:',
                pp.pformat(list(enumerate(Ks))),
                '',
                '- ACTION sofar:',
                pp.pformat(list(enumerate(ACTION))),
            ])

        for i, itm_lks in enumerate(table):
            for itm, lks in itm_lks.items():
                for lk in lks:
                    for ctm, lk1 in G.closure1_with_lookahead(itm, lk):
                        # Try make a REDUCE action for ended item
                        # with lookahead `lk1`
                        if ctm.ended():
                            new_redu = (REDUCE, ctm.r)
                            if lk1 in ACTION[i]:
                                # If there is already an action on
                                # `lk1`, test whether conflicts may
                                # raise against it.
                                act, arg = ACTION[i][lk1]

                                # if the existing action is REDUCE
                                if act is REDUCE:
                                    # which reduces a different item `arg`
                                    if arg != ctm.r:
                                        # then REDUCE/REDUCE conflict is raised.
                                        msg = '\n'.join([
                                            '',
                                            "============================",
                                            '! REDUCE/REDUCE conflict detected',
                                            '- in state {}: '.format(i),
                                            pp.pformat(Ks[i]),
                                            "  on lookahead {}: ".format(repr(lk1)),
                                            '- when creating',
                                            '  {} '.format({lk1: (REDUCE, G.rules[ctm.r])}),
                                            '- against existing',
                                            '  {}'.format({lk1: (REDUCE, G.rules[arg])}),
                                            '',
                                            '- You may need redesign the grammar to avoid proceeding rules',
                                            '  {} and {} into the same state {} and end there.'
                                            .format(G.rules[arg], G.rules[ctm.r], Ks[i]),
                                            '',
                                            intermediate_info(),
                                            "============================",
                                        ])
                                        raise ParserError(msg)

                                # If the existing action is SHIFT
                                else:
                                    # `ctm` is the item prepared for
                                    # shifting on `lk1`
                                    #
                                    # if the left operator
                                    # ctm.rule.rhs[-2] has
                                    # non-strictly higher precedence,
                                    # then revert the SHIFT into
                                    # REDUCE, otherwise preserve the
                                    # SHIFT.

                                    if ctm.size > 1 and ctm.rule.rhs[-2] in G.terminals:
                                        op_lft = ctm.rule.rhs[-2]
                                        op_rgt = lk1
                                        if op_lft in G.precedence and op_rgt in G.precedence:
                                            if G.precedence[op_lft] >= G.precedence[op_rgt]:
                                                # left/REDUCE wins
                                                ACTION[i][lk1] = new_redu
                                            else: # elif G.precedence[op_lft] < G.precedence[op_rgt]:
                                                # right/SHIFT wins
                                                pass
                                        else:
                                            msg = '\n'.join([
                                                '',
                                                "============================",
                                                '! SHIFT/REDUCE conflict detected',
                                                '- in state {}: '.format(i),
                                                pp.pformat(Ks[i]),
                                                "  on lookahead {}: ".format(repr(lk1)),
                                                '- when creating',
                                                '  {} '.format({lk1: (REDUCE, G.rules[ctm.r])}),
                                                '- against existing',
                                                '  {}'.format({lk1: (act, arg)}),
                                                '',
                                                ('- You may need specify precedence values for'
                                                ' both {} and {} if they are designed to be operators.'
                                                 .format(repr(op_lft), repr(op_rgt))),
                                                '',
                                                intermediate_info(),
                                                "============================",
                                            ])
                                            raise ParserError(msg)
                                    else:

                                        # FIXME: how to find the cause of conflict?
                                        # - Propagation path introducing a conflict?

                                        X = ctm.target
                                        problem_rules = []
                                        for rule in G.rules:
                                            if X in rule.rhs:
                                                ix = rule.rhs.index(X)
                                                if -1 < ix < len(rule.rhs) - 1:
                                                    if lk1 in G.first(rule.rhs[ix+1]):
                                                        problem_rules.append(rule)

                                        # propa_chain = [ctm]
                                        # while 1:
                                        #     has_new = False
                                        #     for i, p in enumerate(propa):
                                        #         for (itm, j, jtm) in p:
                                        #             if jtm == propa_chain[0]:
                                        #                 if itm not in propa_chain:
                                        #                     propa_chain.insert(0, itm)
                                        #                     has_new = True
                                        #     if not has_new:
                                        #         break

                                        msg = '\n'.join([
                                            '',
                                            "============================",
                                            '! SHIFT/REDUCE conflict not resolvable by precedence',
                                            '- in state: ',
                                            '  {}'.format(pp.pformat((i, Ks[i]))),
                                            '- when creating',
                                            '  {}'.format({lk1: (REDUCE, ctm)}),
                                            '- against existing',
                                            '  {}'.format({lk1: (act, arg)}),
                                            '',
                                            # '- lookahead table',
                                            # pp.pformat([dict(s) for s in table]),
                                            ('- You may need some redesign to avoid nonterminal {} followed by '
                                             '{} in problematic rules such like:'.format(repr(X), repr(lk1))),
                                            pp.pformat(problem_rules),
                                            # ('- You may redesign the grammar to avoid propagation chain:'),
                                            # pp.pformat(propa_chain),
                                            '',
                                            intermediate_info(),
                                            "============================",
                                        ])
                                        raise ParserError(msg)

                            # If there is still no action on `lk1`.
                            else:
                                ACTION[i][lk1] = (REDUCE, ctm.r)

                # Accept-Item
                if itm.index_pair == (0, 1):
                    ACTION[i][END] = (ACCEPT, itm.r)

        self.ACTION = ACTION

    def parse(self, inp, interp=False, n_warns=5):

        """Perform table-driven deterministic parsing process. Only one parse
        tree is to be constructed.

        If `interp` mode is turned True, then a parse tree is reduced
        to semantic result once its sub-nodes are completed, otherwise
        the parse tree is returned.

        """

        # Aliasing
        L = self.lexer
        Ks = self.Ks
        GOTO = self.GOTO
        ACTION = self.ACTION
        semans = self.semans
        rules = self.rules

        # Tree stack: list of produced subtrees.
        trees = []

        # State stack: list more performant than deque.
        sstack = [0]

        # Lazy extraction of tokens
        toker = L.tokenize(inp, with_end=True) # Use END to force finishing by ACCEPT
        tok = next(toker)
        warns = []

        try:
            while 1:

                # Peek state
                s = sstack[-1]

                if tok.symbol not in ACTION[s]:
                    msg = '\n'.join([
                        '',
                        'WARNING: ',
                        'LALR - Ignoring syntax error reading Token {}'.format(tok),
                        '- Current kernel derivation stack:',
                        pp.pformat([Ks[i] for i in sstack]),
                        '- Expecting tokens and actions:',
                        pp.pformat(ACTION[s]),
                        '- But got: \n{}'.format(tok),
                        '',
                    ])
                    warnings.warn(msg)
                    warns.append(msg)
                    if len(warns) == n_warns:
                        raise ParserError(
                            'Warning tolerance {} reached. Parsing exited.'.format(n_warns))
                    else:
                        tok = next(toker)

                else:
                    act, arg = ACTION[s][tok.symbol]

                    # SHIFT
                    if act == SHIFT:
                        if interp:
                            trees.append(tok.value)
                        else:
                            trees.append(tok)
                        sstack.append(GOTO[s][tok.symbol])
                        # Go on scanning
                        tok = next(toker)

                    # REDUCE
                    elif act == REDUCE:
                        assert isinstance(arg, int)
                        rule = lhs, rhs = rules[arg]
                        seman = semans[arg]
                        subts = deque()
                        for _ in rhs:
                            subt = trees.pop()
                            subts.appendleft(subt)
                            sstack.pop()
                        if interp:
                            tree = seman(*subts)
                        else:
                            tree = ParseTree(rule, subts, seman)
                        trees.append(tree)
                        # New symbol is used for shifting.
                        sstack.append(GOTO[sstack[-1]][lhs])

                    # ACCEPT
                    elif act == ACCEPT:
                        # Reduce the top semantics.
                        assert isinstance(arg, int), arg
                        rule = rules[arg]
                        seman = semans[arg]
                        if interp:
                            return seman(*trees)
                        else:
                            assert len(trees) == 1
                            # Augmented top tree or naive top tree?
                            # return ParseTree(rule, trees, seman)
                            return trees[0]
                    else:
                        raise ParserError('Invalid action {} on {}'.format(act, arg))

        except StopIteration:
            raise ParserError('No enough tokens for completing the parse. ')
            # pass

        return

