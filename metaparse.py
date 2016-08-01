# -*- coding: utf-8 -*-
"""``metaparse``

A tool for powerful instant parsing.

For typical parsing work, a **Python class[1] declaration** will
already suffice. This pseudo-class includes

-  lexical definition
-  rule definition
-  semantic definition

all-in-one for a grammar.


Example:
.. code:: python

    from metaparse import cfg, LALR

    # Global stuff
    table = {}

    class G_Calc(metaclass=cfg):

        # ===== Lexical patterns / Terminals =====

        IGNORED = r'\s+'            # Special token.

        EQ  = r'='
        NUM = r'[0-9]+'
        ID  = r'[_a-zA-Z]\w*'
        POW = r'\*\*', 3            # Can specify token precedence (mainly for LALR).
        MUL = r'\*'  , 2
        ADD = r'\+'  , 1

        # ===== Syntactic/Semantic rules in SDT-style =====

        def assign(ID, EQ, expr):        # May rely on global side-effects...
            table[ID] = expr

        def expr(NUM):                   # or return local results for purity.
            return int(NUM)

        def expr(expr_1, ADD, expr_2):   # With TeX-subscripts, meaning (expr → expr₁ + expr₂).
            return expr_1 + expr_2

        def expr(expr, MUL, expr_1):     # Can ignore one of the subscripts.
            return expr * expr_1

        def expr(expr, POW, expr_1):
            return expr ** expr_1

    Calc = LALR(G_Calc)


Usage:
.. code:: python

    >>> Calc.interpret("x = 1 + 2 * 3 ** 4 + 5")
    >>> Calc.interpret("y = 3 ** 4 * 5")
    >>> Calc.interpret("z = 99")

    >>> table
    {'x': 168, 'z': 99, 'y': 405}


TODO:
    - Support GSS structure for non-deterministic parsing.
    - Online-parsing


INFO:
    Author:  ShellayLee
    Mail:    shellaylee@hotmail.com
    License: MIT


.. _Project site:
    https://github.com/Shellay

"""

import re
import warnings
import inspect
import textwrap
import ast
import pprint as pp
import json
import pickle

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
END_PATTERN_DEFAULT = r'$'

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

START   = Symbol('START')
PREDICT = Symbol('PREDICT')
SHIFT   = Symbol('SHIFT')
REDUCE  = Symbol('REDUCE')
ACCEPT  = Symbol('ACCEPT')


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

    """Rule object has the form (LHS -> RHS).  It is mainly constructed by
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
        # Signature only works in Python 3
        # for x in inspect.signature(func).parameters:
        for x in inspect.getargspec(func).args:
            # Tail digital subscript like xxx_4
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


    Notes:

    * Auto-augmentation: add a singleton top rule if not explicitly
      given.

    * Auto-check:

        * Reference to undeclared symbols;
        * Unused tokens;
        * Unreachable rules;
        * LOOPs;

    """

    def __init__(self, lex2pats, rule2semans, prece=None, lexhdls=None):
        """
        Parameters:

        :lexes:  : list<(<terminal-name> : <terminal-pattern>)>
            A ordered dict representing lexical rules.

        :rules:  : [Rule]
            A list of grammar rules.

        :prece:  : {<token>: <int-precedence>}
            A dict of operator's precedence numbers.

        """

        # Operator precedence table may be useful.
        self.OP_PRE = dict(prece) if prece else {}
        self.lex_handlers = dict(lexhdls) if lexhdls else {}

        # Cache terminals
        self.terminals = set()
        self.lex2pats = []
        for tmn, pat in lex2pats:
            # Name trailing with integer subscript indicating
            # the precedence.
            self.terminals.add(tmn)
            # self.lex2pats.append((tmn, re.compile(pat, re.MULTILINE)))
            self.lex2pats.append((tmn, pat))

        # Cache nonterminals
        self.nonterminals = []
        self.rules = []
        self.semans = []
        for rule, seman in rule2semans:
            if rule.lhs not in self.nonterminals:
                self.nonterminals.append(rule.lhs)
            self.rules.append(rule)
            self.semans.append(seman)

        # Sort rules with consecutive grouping of LHS.
        # rules.sort(key=lambda r: self.nonterminals.index(r.lhs))

        # This block checks completion of given grammar.
        unused_t = set(self.terminals).difference([IGNORED, END, ERROR])
        unused_nt = set(self.nonterminals).difference([self.nonterminals[0]])
        # Raise grammar error in case any symbol is undeclared
        msg = ''
        for r, rl in enumerate(self.rules):
            for j, X in enumerate(rl.rhs):
                unused_t.discard(X)
                unused_nt.discard(X)
                if X not in self.terminals and X not in self.nonterminals:
                    msg += '\n'.join([
                        '',
                        '====================',
                        'Undeclared symbol:',
                        "@{}`th symbol '{}' in {}`th rule {}. Source info:".format(j, X, r, rl),
                        rl.src_info,
                        '====================',
                        '',
                    ])
        if msg:
            msg += '\n...Failed to construct the grammar.'
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
        fst_rl = self.rules[0]
        if len(fst_rl.rhs) != 1 or 1 < [rl.lhs for rl in rules].count(fst_rl.lhs):
            tp_lhs = "{}^".format(fst_rl.lhs)
            tp_rl = Rule(tp_lhs, (fst_rl.lhs,))
            self.nonterminals.insert(0, tp_lhs)
            self.rules = [tp_rl] + self.rules
            self.semans = [id_func] + self.semans

        self.symbols = self.nonterminals + [a for a in self.terminals if a != END]

        # Top rule of Augmented Grammar.
        self.top_rule = self.rules[0]
        self.top_symbol = self.top_rule.lhs

        # Helper for fast accessing with Trie like structure.
        self._ngraph = defaultdict(list)
        for r, rl in enumerate(self.rules):
            self._ngraph[rl.lhs].append((r, rl))

        # Prepare useful information for parsing.
        self._calc_nullable()
        self._calc_pred_trees()

    def __repr__(self):
        return 'Grammar\n{}\n'.format(pp.pformat(self.rules))

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
                # assert G.is_nullable_seq(rl.rhs)
                # assert isinstance(nxt, ExpdNode)
                # #
                # for fk in nxt.forks:
                #     # `fk` maybe the bottom.
                #     if fk is not bottom:
                #         (act1, arg), _ = fk
                #         if act1 is PREDICT:
                #             toplst.append(fk)
                #         elif act1 is REDUCE:
                #             if fk not in toplst:
                #                 toplst.append(fk)
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
        G.FIRST = {}
        for nt in reversed(G.nonterminals):
            pt = G.pred_tree(nt)
            G.FIRST[nt] = f = set()
            for nd in pt:
                if isinstance(nd, Node):
                    act, x = nd.value
                    if act is PREDICT:
                        assert x in G.terminals
                        f.add(x)
                    elif act is REDUCE:
                        assert x in G.rules
                        f.add(EPSILON)
                    else:
                        raise ValueError('Not valid action in toplist.')
                else:
                    raise ValueError('Not valid node in toplist.')

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

    def __init__(self, lex2pats, lex_handlers):
        # Raw info
        self.extend(lex2pats)
        # Compiled to re object
        self.lex2rgxs = [
            (lex, re.compile(pat))
            for lex, pat in lex2pats
        ]
        # Handlers
        self.lex_handlers = lex_handlers
        
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

        pos = 0
        while pos < len(inp):
            m = None
            for cat, rgx in self.lex2rgxs:
                m = rgx.match(inp, pos=pos)
                # The first match with non-zero length is yielded.
                if m and len(m.group()) > 0:
                    break
            if m:
                if cat == IGNORED:
                    # Need IGNORED handler?
                    at, pos = m.span()
                elif cat == ERROR:
                    # Call ERROR handler!
                    at, pos = m.span()
                    lxm = m.group()
                    if ERROR in self.lex_handlers:
                        h = self.lex_handlers[ERROR]
                        yield Token(at, ERROR, lxm, h(lxm))
                    else:
                        yield Token(at, ERROR, lxm, lxm)
                else:
                    at, pos = m.span()
                    lxm = m.group()
                    if cat in self.lex_handlers:
                        # Call normal token handler.
                        h = self.lex_handlers[cat]
                        yield Token(at, cat, lxm, h(lxm))
                    else:
                        yield Token(at, cat, lxm, lxm)
            else:
                # Report unrecognized Token here!
                raise GrammarError('No defined pattern for unrecognized symbol {}!'.format(inp[at]))
        if with_end:
            yield Token(pos, END, END_PATTERN_DEFAULT, None)

    def to_json(self, indent=4, **kw):
        lex2pats = self[:]
        hdls = {}
        for lex, hdl in self.lex_handlers.items():
            # FIXME: inspect.getsource may fail for loaded object!
            hdls[lex] = textwrap.dedent(inspect.getsource(hdl))
        obj = {'lex2pats': lex2pats, 'lex_handlers': hdls}
        return json.dumps(obj, indent=indent, **kw)

    @staticmethod
    def from_json(s, glb_ctx=None):
        obj = json.loads(s)
        lex2pats = obj['lex2pats']
        lexhdls = {}
        ctx = {}
        for lex, src in obj['lex_handlers'].items():
            co = compile(src, '<string>', 'exec')
            if glb_ctx:
                exec(co, glb_ctx, ctx)
            else:
                exec(co, {}, ctx)
            lexhdls[lex] = ctx[lex]
        return Lexer(lex2pats, lexhdls)


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

                    # Scheme 1:
                    # '''
                    # def ID(lex: r'\w\w') -> 3:
                    #     return lex
                    # '''
                    if v.__name__ in lexes:
                        # assert 'lex' in v.__annotations__, 'Must declare param `lex`.'
                        assert inspect.getargs(v.__code__).args == ['lex'], \
                            "Lexical handler must have singleton param list ['lex']"
                        lexhdls[v.__name__] = v
                    elif hasattr(v, '__annotations__') and 'lex' in v.__annotations__:
                        lx = v.__name__
                        ann = v.__annotations__
                        assert v.__code__.co_argcount == 1, 'Single argument handler required.'
                        lexes.append(lx)
                        lexpats.append(ann['lex'])
                        lexhdls[lx] = v
                        if 'return' in ann:
                            prece[lx] = ann['return']

                    # For rule declarations
                    else:
                        r = Rule.read(v)
                        if r in rules:
                            raise GrammarError('Repeated declaration of Rule {}.\n{}'.format(r))
                        rules.append(r)
                        semans.append(v)

                # Lexical declaration allows repeatition!!
                # - String pattern without precedence
                elif isinstance(v, str):
                    lexes.append(k)
                    lexpats.append(v)
                # - String pattern with precedence
                elif isinstance(v, tuple):
                    assert len(v) == 2, 'Lexical pattern as tuple should be of length 2.'
                    assert isinstance(v[0], str) and isinstance(v[1], int), \
                        'Lexical pattern as tuple should be of type (str, int)'
                    lexes.append(k)
                    lexpats.append(v[0])
                    prece[k] = v[1]

                # Handle normal private attributes/methods.
                else:
                    attrs.append((k, v))

        # Default matching order of special patterns:

        # Always match IGNORED secondly after END, if it is not specified;
        if IGNORED not in lexes:
            # lexes.move_to_end(IGNORED, last=False)
            lexes.append(IGNORED)
            lexpats.append(IGNORED_PATTERN_DEFAULT)

        # Always match END first
        if END not in lexes:
            lexes.insert(0, END)
            lexpats.insert(0, END_PATTERN_DEFAULT)
        else:
            i_e = lexes.index(END)
            lexes.insert(0, lexes.pop(i_e))
            lexpats.insert(0, lexpats.pop(i_e))

        # Always match ERROR at last
        # It may be overriden by the user.
        if ERROR not in lexes:
            # lexes[ERROR] = ERROR_PATTERN_DEFAULT
            lexes.append(ERROR)
            lexpats.append(ERROR_PATTERN_DEFAULT)

        g = Grammar(zip(lexes, lexpats), zip(rules, semans), prece, lexhdls)
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
                # :ast.Module: is the unit of program codes.
                # FIXME: Is :md: necessary??
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

        # Example:
        # - prediction stack
        # - processed argument stack

        # i-  [T]$
        # a- $[]

        # =(T -> AB)=>>

        # i-  [A B (->T)]$
        # a- $[]

        # =(A -> a1 a2)=>>

        # i-  [a1 a2 (->A) B (->T)]$
        # a- $[]

        # =a1, a2=>>

        # i-  [(->A) B (->T)]$
        # a- $[a1 a2]

        # =(a1 a2 ->A)=>>

        # i-  [B (->T)]$
        # a- $[A]

        # =(B -> b)=>>

        # i-  [b (->B) (->T)]$
        # a- $[A]

        # =b=>>

        # i-  [(->B) (->T)]$
        # a- $[A b]

        # =(b ->B)=>>

        # i-  [(->T)]$
        # a- $[A B]

        # =(A B ->T)=>>

        # i-  []$
        # a- $[T]

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
        assert isinstance(grammar, Grammar)
        self.lexer = Lexer.from_grammar(grammar)

    def __repr__(self):
        # Polymorphic representation without overriding
        # raise NotImplementedError('Parser should override __repr__ method. ')
        return self.__class__.__name__ + '-Parser-{}'.format(self.grammar)

    def parse_many(self, inp, interp=False):
        # Must be overriden
        raise NotImplementedError('Any parser should have a parse method.')

    def interpret_many(self, inp):
        return self.parse_many(inp, interp=True)

    def parse(self, inp, interp=False):
        # res = self.parse_many(inp, interp)
        # if res:
        #     return res[0]
        # else:
        #     raise ParserError("No parse.")
        raise NotImplementedError('For non-deterministic parsers use :parse_many: instead.')

    def interpret(self, inp):
        # return self.parse(inp, interp=True)
        raise NotImplementedError('For non-deterministic parsers use :interpret_many: instead.')


class ParserDeterm(ParserBase):

    """Abstract class for deterministic parsers. While the *parse_many* method
    tends to return a list of results and deterministic parsers yield
    at most ONE result, *parse* and *interpret* return this result
    directly.

    """
    # def __init__(self, grammar):

    def parse_many(self, inp, interp=False):
        return [self.parse(inp, interp=interp)]

    def interpret_many(self, inp):
        return [self.interpret(inp)]

    def parse(self, inp, interp):
        raise NotImplementedError('Deterministic parser should have parse1 method.')

    def interpret(self, inp):
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
                    msg = '\n===================='
                    msg += '\nUnrecognized {}'.format(tok)
                    msg += '\nChoked active ItemSet: \n{}\n'.format(pp.pformat(C))
                    msg += '====================\n'
                    msg += '\nStates:'
                    msg += '\n{}\n'.format(pp.pformat(S))
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


# TODO:
# 
# LR parsers can indeed corporate pre-order semantics. Each ItemSet
# has prepared a set of prediction paths, while each GOTO action
# confirms which path to go and triggers the corresponding semantics
# (though at most ONE token gets consumed already).
# 
@meta
class GLR(ParserBase):

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

    def __init__(self, grammar=None, dict=None):
        # super(GLR, self).__init__(grammar)
        if grammar is not None:
            self.lexer = Lexer.from_grammar(grammar)
            self._build_automaton(grammar)
        elif dict is not None:
            for k, v in dict.items():
                setattr(self, k, v)

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

        self.rules = G.rules[:]
        self.semans = G.semans[:]

        # Kernels
        self.Ks = Ks = [[G.item(0, 0)]]
        self.GOTO = GOTO = []
        self.ACTION = ACTION = []

        # Construct LR(0)-DFA
        k = 0
        while k < len(Ks):

            I = Ks[k]
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

            # igoto = OrderedDict()
            igoto = {}
            for X, J in igotoset.items():
                # J = sorted(J, key=lambda i: (i.r, i.pos))
                J.sort()
                if J not in Ks:
                    Ks.append(J)
                j = Ks.index(J)
                iacts[SHIFT][X] = j
                igoto[X] = j

            ACTION.append(iacts)
            GOTO.append(igoto)

            k += 1

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
                            results.append(subs[0])
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

    def dumps(self):
        obj = dict(
            lexer = self.lexer.to_json(),
            semans = [textwrap.dedent(inspect.getsource(seman))
                      for seman in self.semans],
            rules = self.rules,
            Ks = self.Ks,
            ACTION = self.ACTION,
            GOTO = self.GOTO,
        )
        s = pickle.dumps(obj)
        return s

    @staticmethod
    def loads(s, glb_ctx={}):
        obj = pickle.loads(s)
        obj['lexer'] = Lexer.from_json(obj['lexer'], glb_ctx)
        semans = []
        for sm_src in obj['semans']:
            ctx = {}
            co = compile(sm_src, '<string>', 'exec')
            exec(co, glb_ctx, ctx)
            k, sm = ctx.popitem()
            semans.append(sm)
        obj['semans'] = semans
        return GLR(dict=obj)

@meta
class LALR(ParserDeterm):

    """Look-Ahead Leftmost-reading Rightmost-derivation (LALR) parser may
    be the most widely used parser variant. It has almost the same
    automaton like GLR, with potential ambiguity eliminated by raising
    SHIFT/REDUCE and REDUCE/REDUCE conflicts.

    Due to its deterministic nature and table-driven process does it
    have linear-time performance.

    """

    def __init__(self, grammar=None, dict=None):

        if grammar is not None:
            self._build_automaton(grammar)

            self.lexer = Lexer.from_grammar(grammar)
            self.rules = grammar.rules[:]
            self.semans = grammar.semans[:]
        elif dict is not None:
            for k, v in dict.items():
                setattr(self, k, v)

    def __repr__(self):
        return 'LALR-parser for {}'.format(pp.pformat(self.rules))

    def _build_automaton(self, G):

        Ks = [[G.item(0, 0)]]   # Kernels
        GOTO = []

        # Calculate LR(0) item sets and GOTO
        i = 0
        while i < len(Ks):

            K = Ks[i]

            # Use OrderedDict to preserve order of found goto's.
            igoto = OrderedDict()

            # SetOfItems algorithm
            for itm in G.closure(K):
                # If item (A -> α.Xβ) has a goto.
                if not itm.ended():
                    X, jtm = itm.actor, itm.shifted
                    if X not in igoto:
                        igoto[X] = []
                    if jtm not in igoto[X]:
                        igoto[X].append(jtm)

            # Register `igoto` into global goto.
            GOTO.append({})
            for X, J in igoto.items():
                # The Item-Sets should be treated as UNORDERED! So
                # sort J to identify the Lists with same items.
                # J = sorted(J, key=lambda i: (i.r, i.pos))
                J.sort()
                if J not in Ks:
                    Ks.append(J)
                j = Ks.index(J)
                GOTO[i][X] = j

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
                            # Propagation from KERNEL item :ktm: to
                            # its belonging non-kernel item :ctm:,
                            # which is shifted into :j:'th item set
                            # (by its actor). See ALGO 4.62 in Dragon
                            # book.
                            propa[i].append((ktm, j, ctm.shifted))

        table = spont

        # MORE-PASS propagation
        b = 1
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
                b += 1

        # self._propa_passes = b

        self.Ks = Ks
        self.GOTO = GOTO
        # self.propa = propa
        # self.table = table

        # Now all goto and lookahead information is available.

        # Construct ACTION table
        ACTION = [{} for _ in table]

        # SHIFT for non-ended items
        # Since no SHIFT/SHIFT conflict exists, register
        # all SHIFT information firstly.
        for i, xto in enumerate(GOTO):
            for X, j in xto.items():
                if X in G.terminals:
                    ACTION[i][X] = (SHIFT, j)

        # REDUCE for ended items
        # SHIFT/REDUCE and REDUCE/REDUCE conflicts are
        # to be found.
        conflicts = []
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
                                # `lk1`, test wether conflicts may
                                # raise for it.
                                act, arg = ACTION[i][lk1]
                                # if the action is REDUCE
                                if act is REDUCE:
                                    # which reduces a different item `arg`
                                    if arg != ctm:
                                        # then REDUCE/REDUCE conflict is raised.
                                        conflicts.append((i, lk1, (act, arg), new_redu))
                                        continue
                                # If the action is SHIFT
                                else:
                                    # `ctm` is the item prepared for
                                    # shifting on `lk`
                                    #
                                    # if the left operator
                                    # ctm.rule.rhs[-2] has
                                    # non-strictly higher precedence,
                                    # then revert the SHIFT into
                                    # REDUCE, else preserve the SHIFT
                                    if lk1 in G.OP_PRE:
                                        if ctm.size > 1 and ctm.rule.rhs[-2] in G.OP_PRE:
                                            op_lft = ctm.rule.rhs[-2]
                                            op_rgt = lk1
                                            # Trivially, maybe `op_lft` == `op_rgt`
                                            if G.OP_PRE[op_lft] >= G.OP_PRE[op_rgt]:
                                                # Left wins.
                                                ACTION[i][lk1] = new_redu
                                                continue
                                            else:
                                                # Right wins.
                                                continue
                                    # SHIFT/REDUCE conflict raised.
                                    conflicts.append((i, lk1, (act, arg), (REDUCE, ctm.r)))
                            # If there is still no action on `lk1`.
                            else:
                                ACTION[i][lk1] = (REDUCE, ctm.r)
                # Accept-Item
                if itm.index_pair == (0, 1):
                    ACTION[i][END] = (ACCEPT, itm.r)

        # Report LALR-conflicts, if any.
        if conflicts:
            msg = "\n============================"
            for i, lk, act0, act1 in conflicts:
                msg += '\n'.join([
                    '',
                    '! LALR-Conflict raised:',
                    '  * in state [{}]: '.format(i),
                    pp.pformat(Ks[i]),
                    "  * on lookahead {}: ".format(repr(lk)),
                    pp.pformat({lk: [act0, act1]}),
                    '',
                ])
            msg += "============================"
            raise ParserError(msg)

        self.ACTION = ACTION

    def parse(self, inp, interp=False, n_warns=5):

        """Perform table-driven deterministic parsing process. Only one parse
        tree is to be constructed.

        If `interp` mode is turned True, then a parse tree is reduced
        to semantic result once its sub-nodes are completed, otherwise
        the parse tree is returned.

        """

        # Aliasing
        trees = []
        L = self.lexer
        Ks = self.Ks
        GOTO = self.GOTO
        ACTION = self.ACTION
        semans = self.semans
        rules = self.rules

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
                        rl = rules[arg]
                        seman = semans[arg]
                        subts = deque()
                        for _ in range(rl.size):
                            subt = trees.pop()
                            subts.appendleft(subt)
                            sstack.pop()
                        if interp:
                            tree = seman(*subts)
                        else:
                            tree = ParseTree(rl, subts, seman)
                        trees.append(tree)
                        # New symbol is used for shifting.
                        sstack.append(GOTO[sstack[-1]][rl.lhs])

                    # ACCEPT
                    elif act == ACCEPT:
                        # Reduce the top semantics.
                        assert isinstance(arg, int), arg
                        rl = rules[arg]
                        seman = semans[arg]
                        if interp:
                            return seman(*trees)
                        else:
                            return ParseTree(rl, trees, seman)
                    else:
                        raise ParserError('Invalid action {} on {}'.format(act, arg))

        except StopIteration:
            raise ParserError('No enough tokens for completing the parse. ')
            # pass

        return

    def dumps(self):
        obj = dict(
            lexer = self.lexer.to_json(),
            semans = [textwrap.dedent(inspect.getsource(seman))
                      for seman in self.semans],
            rules = self.rules,
            Ks = self.Ks,
            ACTION = self.ACTION,
            GOTO = self.GOTO,
        )
        s = pickle.dumps(obj)
        return s

    @staticmethod
    def loads(s, glb_ctx={}):
        obj = pickle.loads(s)
        obj['lexer'] = Lexer.from_json(obj['lexer'], glb_ctx)
        semans = []
        for sm_src in obj['semans']:
            ctx = {}
            co = compile(sm_src, '<string>', 'exec')
            exec(co, glb_ctx, ctx)
            k, sm = ctx.popitem()
            semans.append(sm)
        obj['semans'] = semans
        return LALR(dict=obj)


@meta
class WLL1(ParserDeterm):
    """Weak-LL(1)-Parser.

    Since 'strong'-LL(1) grammar parser includes the usage of FOLLOW
    set, which is only heuristically helpful for the recognitive
    capability when handling NULLABLE rules, this parser suppress the
    need of FOLLOW.

    When deducing a NULLABLE nonterminal A with some lookahead a, if a
    does not belong to any FIRST of A's alternatives, then the NULL
    alternative is chosen. In other words, all terminals not in
    FIRST(A) leads to the prediction (as well as immediate reduction)
    of (A -> ε) in the predictive table.

    This variation allows predicting (A -> ε) even when lookahead a is
    not in FOLLOW, which means this parser will postpone the
    recognition error compared to strong-LL(1) parser.

    """

    def __init__(self, grammar):
        super(WLL1, self).__init__(grammar)
        self._calc_ll1_table()

    def _calc_ll1_table(self):
        G = self.grammar
        table = self.table = {}
        for r, rule in enumerate(G.rules):
            lhs, rhs = rule
            if lhs not in table:
                table[lhs] = {}
            # NON-NULL rule
            if rhs:
                for a in G.first_of_seq(rhs, EPSILON):
                    if a is EPSILON:
                        pass
                    elif a in table[lhs]:
                        raise GrammarError('Not simple LL(1) grammar! ')
                    else:
                        table[lhs][a] = rule
            # NULL rule
            # This rule tends to be tried when
            # the lookahead doesn't appear in
            # other sibling rules.
            else:
                pass

    def parse(self, inp, interp=False):
        """The process is exactly the `translate' process of a ParseTree.

        """
        # Backtracking is yet supported
        # Each choice should be deterministic
        push = list.append
        pop = list.pop
        G = self.grammar
        pstack = self.pstack = []
        table = self.table
        toker = enumerate(G.tokenize(inp, with_end=True))
        pstack.append(G.rules[0].lhs)
        argstack = []
        try:
            k, tok = next(toker)
            while pstack:
                actor = pop(pstack)
                at, look, tokval = tok
                # Reduction
                if isinstance(actor, Rule):
                    args = []
                    # Pop the size of args, conclude subtree
                    # for prediction made before
                    for _ in actor.rhs:
                        args.insert(0, pop(argstack))
                    if interp:
                        arg1 = actor.seman(*args)
                    else:
                        arg1 = ParseTree(actor, args)
                    # Finish - no prediction in stack
                    # Should declare end-of-input
                    if not pstack:
                        return arg1
                    else:
                        push(argstack, arg1)
                # Make prediction on nonterminal
                elif actor in G.nonterminals:
                    if look in table[actor]:
                        pred = table[actor][look]
                        # Singal for reduction
                        push(pstack, pred)
                        # Push symbols into prediction-stack,
                        # last symbol first in.
                        for x in reversed(pred.rhs):
                            push(pstack, x)
                    # !!! Heuristically do epsilon-reduction when no
                    # viable lookahead found
                    elif actor in G.NULLABLE:
                        for r0 in G.rules:
                            if r0.lhs == actor and not r0.rhs:
                                if interp:
                                    argstack.append(r0.seman())
                                else:
                                    argstack.append(ParseTree(r0, []))
                    # Recognition failed, ignore
                    else:
                        raise ParserError('No production found.')
                # Try match terminal
                else:
                    if actor == look:
                        if interp:
                            argstack.append(tokval)
                        else:
                            argstack.append(tok)
                    k, tok = next(toker)

        except StopIteration:
            raise ParserError('No enough tokens to complete parsing.')


@meta
class GLL(ParserBase):
    """This implemenation of GLL parser can handle left-recursion,
    left-sharing problem and loops properly. However, *hidden left
    recursion* must be handled with special caution and is currently
    not supported.

    """

    def __init__(self, grammar):
        # super(GLL, self).__init__(grammar)
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

        stk_btm = START
        # `bottom` comes after the reduction of top-symbol
        bottom = ExpdNode(ACCEPT, [])

        # GSS push
        push = Node

        # (<active-node>, <cumulative-stack>)
        toplst = [(n, stk_btm)
                  for n in G.pred_tree(G.top_symbol, bottom)]
        results = []

        for k, tok in enumerate(G.tokenize(inp, with_end=True)):

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
                        # if m not in expdd:
                        if 1:
                            # expdd[m] = nxt
                            for fk in fks:
                                # srchs.append((fk, stk + [x, nxt.value]))
                                # Mind the push order!
                                # - ExpdNode redundant in stack, thus not pushed.
                                srchs.append((fk, {**expdd, m: nxt}, push(rl, stk)))

            toplst = toplst1

        return results
