"""==============================
        metaparse.py
==============================


* Intro and Highlights

This module prepares alternative utilities for syntactical
analyzing in pure Python environment. It includes:

    * OO-style grammar object system for context-free languages with
      completeness check.

    * Elegant grammar definer based upon metaprogramming.

    * Parser interface.

    * Optional parsing algorithms (Earley, GLR, LALR etc.).

To allow the ultimate ease of usage, the OO-style grammar definition
is approached by the Python class structure by treating the method
definitions as syntactic rules associated with semantic behaviors.


* Example

For a left-right-value grammar in C-style language::

        '''
        S -> L = R
        S -> R
        L -> * R
        L -> id
        R -> L
        '''

In Python >= 3, a handy LALR parser for this grammar based on this
module can be written as::


    class P_LRVal(metaclass=LALR.meta):

        IGNORED = r'\s+'
        EQ   = r'='
        STAR = r'\*'
        ID   = r'\w+'

        def S(L, EQ, R) : return ('assign', L, R)
        def S(R)        : return ('expr', R)
        def L(STAR, R)  : return ('deref', R)
        def L(ID)       : return ID
        def R(L)        : return L


The usage is highly straightforward::

    >>> P_LRVal.interpret('abc')
    ('expr', 'abc')

    >>> P_LRVal.interpret('abc = * * ops')
    ('assign', 'abc', ('deref', ('deref', 'ops')))

    >>> P_LRVal.interpret('* abc = * * * ops')
    ('assign', ('deref', 'abc'), ('deref', ('deref', ('deref', 'ops'))))


* Paraphrase

It may seem unusual but interesting that such sort of a method
declaration can play two roles at the same time, one is the formal
syntactic rule represented by the signature literals, while the other
is the semantic behavior interpreting the rule in the Python runtime
environment.

By applying the metaclass, the original behavior of Python class
declaration is overriden (this style of using metaclass is only
available in Python 3.X), which has the following new meanings:


    Attribute declarations

        * LHS is the name of the Token (lexical unit)

        * RHS is the pattern of the Token, which obeys the Python regular
        expression syntax (see documentation of the `re` module)

        * The order of declarations matters. Since there may be patterns
        that overlap, the patterns in prior positions are matched first
        during tokenizing


    Class level method declarations

        * Method name is the rule-LHS, i.e. nonterminal symbol

        * Method paramter list is the rule-RHS, i.e. a sequence of
        symbols. Moreover, each parameter binds to the successful
        subtree or result of executing the subtree's semantic rule
        during parsing the symbol

        * Method body specifies the semantic behavior associated with the
        rule. The returned value is treated as the result of successfully
        parsing input with this rule


* Notes

A limitation is that certain symbols reserved for Python are not usable
to declare grammar in OO-style

Although such representation of grammar is somewhat verbose especially
by the definition of alternative productions, it clearly specifies
alternative semantic behaviors with NAMED parameters, which appears to
be more expressive than POSITIONAL parameters, like %1, %2 ... in
YACC/BISON tool-sets.

"""

import re
import warnings
import inspect
import pprint as pp

from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple
from collections import deque


class GrammarWarning(UserWarning):
    pass


class GrammarError(Exception):
    pass


class ParserWarning(UserWarning):
    pass


class ParserError(Exception):
    pass


# Special lexical element and lexemes
END = 'END'
END_PATTERN_DEFAULT = r'\Z'

IGNORED = 'IGNORED'
IGNORED_PATTERN_DEFAULT = r'[ \t]'

ERROR = 'ERROR'
ERROR_PATTERN_DEFAULT = r'.'

DUMMY = '\0'

class _EPSILON:
    """A singleton object representing empty productivity of nonterminals."""

    def __repr__(self):
        return 'EPSILON'


EPSILON = _EPSILON()

# # Object Token
# Token = namedtuple('Token', 'at symbol value')
# Token.start = property(lambda s: s.at)
# Token.end = property(lambda s: s.at + len(s.value))
# Token.__repr__ = lambda s: '({} -> {})@[{}:{}]'.format(s.symbol, repr(s.value), s.at, s.end)
# Token.is_END = lambda s: s.symbol == END


class Token(object):

    def __init__(self, at, symbol, value):
        self.at = at
        self.symbol = symbol
        self.value = value

    def __repr__(self):
        return '({} -> {})@[{}:{}]'.format(
            self.symbol,
            repr(self.value),
            self.at,
            self.at + len(self.value))

    def __eq__(self, other):
        return self.symbol == other.symbol

    def __iter__(self):
        yield self.at
        yield self.symbol
        yield self.value

    @property
    def start(self):
        return self.at

    @property
    def end(self):
        return self.at + len(self.value)


class Rule(object):

    """Rule object representing a rule in a Context Free Grammar. A Rule
    inside some specified scope should be constructed only once. The
    equality should not be overwritten, thus as identity to save
    performance when comparing rules while comparing some Item
    objects.

    The rule is mainly for traditional canonical BNF
    representation. For grammar systems like EBNF based upon Parsing
    Expression Grammar, which supports
    Star/Optional/Alternative/... modifiers, it is better to define a
    separate object for the rule-like expressions, although it is a
    super set of canonical BNF rule.

    """

    def __init__(self, func):
        self.lhs = func.__name__
        self.rhs = [x[:-1] if x[-1].isdigit() else x
                    for x in inspect.signature(func).parameters]
        self.rhs = []
        for x in inspect.signature(func).parameters:
            # Tail digital subscript like xxx_4
            s = re.search(r'_(\d+)$', x)
            # s = re.search(r'_?(\d+)$', x)
            if s:
                x = x[:s.start()]
            self.rhs.append(x)
        self.seman = func
        self.anno = func.__annotations__

    def __repr__(self):
        return '( {} -> {} )'.format(self.lhs, ' '.join(self.rhs))

    def __iter__(self):
        yield self.lhs
        yield self.rhs

    @staticmethod
    def raw(lhs, rhs, seman):
        rl = Rule(seman)
        rl.lhs = lhs
        rl.rhs = list(rhs)
        return rl


class Item(object):

    """Item contains a pointer to rule list, a index of rule within that
    list and a position indicating current read state in this Item.

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
        return '({} -> {}.{})'.format(lhs, rhs1, rhs2)

    def __eq__(self, x):
        return self.r == x.r and self.pos == x.pos

    def __hash__(self):
        return hash((self.r, self.pos))

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

    def eval(self, *args):
        return self.rule.seman(*args)


class Grammar(object):

    def __init__(self, lexes: dict, rules: list, attrs: list):
        """
        Parameters:

        :lexes:  : odict{str<terminal-name> : str<terminal-pattern>}
            A ordered list of pairs representing lexical rules.

        :rules:  : [Rule]
            A list of grammar rules.

        Notes:

        * Checks whether there is a singleton TOP-rule
          and add such one if not.

        * Needs to check validity and completeness!!
            * Undeclared tokens;
            * Undeclared nonterminals;
            * Unused tokens;
            * Unreachable nonterminals/rules;
            * Cyclic rules;

        """

        # odict{lex-name: lex-re}
        self.terminals = OrderedDict()
        for tmn, pat in lexes.items():
            self.terminals[tmn] = re.compile(pat, re.MULTILINE)

        # [<unrepeated-nonterminals>]
        self.nonterminals = []
        for rl in rules:
            if rl.lhs not in self.nonterminals:
                self.nonterminals.append(rl.lhs)

        # This block checks completion of given grammar.
        _t_unused = set(self.terminals).difference([IGNORED, END, ERROR])
        _nt_unused = set(self.nonterminals).difference([rules[0].lhs])

        # Raise grammar error in case any symbol is undeclared
        msg = ''
        for r, rl in enumerate(rules):
            for j, X in enumerate(rl.rhs):
                _t_unused.discard(X)
                _nt_unused.discard(X)
                if X not in self.terminals and X not in self.nonterminals:
                    msg += '\nUndeclared symbol:'
                    msg += "@{}`th symbol '{}' in {}`th rule {}.".format(j, X, r, rl)
        if msg:
            msg += '\n...Failed to construct the grammar.'
            raise GrammarError(msg)

        # Raise warnings in case any symbol is unused.
        for t in _t_unused:
            # warnings.warn('Warning: referred terminal symbol {}'.format(t))
            warnings.warn('Unreferred terminal symbol {}'.format(repr(t)))
        for nt in _nt_unused:
            # warnings.warn('Warning: referred nonterminal symbol {}'.format(nt))
            warnings.warn('Unreferred nonterminal symbol {}'.format(repr(nt)))


        # Generate top rule as Augmented Grammar only if
        # the top rule is not explicitly given.
        fst_rl = rules[0]
        if len(fst_rl.rhs) != 1 or 1 < [rl.lhs for rl in rules].count(fst_rl.lhs):
            # For LR(1):
            # - END should not be considered as a symbol in grammar,
            #   only as a hint for acceptance;
            # For LR(0):
            # - END should be added into grammar.
            tp_lhs = "{}^".format(fst_rl.lhs)
            tp_rl = Rule.raw(
                tp_lhs,
                (fst_rl.lhs, ),
                lambda x: x)
            self.nonterminals.insert(0, tp_lhs)
            rules = [tp_rl] + rules

        # Register other attributes/methods
        self.attrs = attrs

        self.rules = rules
        self.symbols = self.nonterminals + [a for a in lexes if a != END]

        # Top rule of Augmented Grammar.
        self.top_rule = self.rules[0]
        self.top_symbol = self.top_rule.lhs

        # Helper for fast accessing with Trie like structure.
        self.ngraph = defaultdict(list)
        for rl in self.rules:
            self.ngraph[rl.lhs].append(rl)

        # Prepare useful information for parsing.
        self._calc_first_and_nullable()

    def __repr__(self):
        return 'Grammar\n{}\n'.format(pp.pformat(self.rules))

    def __getitem__(self, k: str):
        if k in self.ngraph:
            return self.ngraph[k]
        else:
            raise ValueError('No such LHS {} in grammar.'.format(k))

    def make_item(G, r: int, pos: int):
        """Create a pair of integers indexing the rule and active position.

        """
        return Item(G.rules, r, pos)

    def _calc_first_and_nullable(G):
        """Calculate the FIRST set of this grammar as well as the NULLABLE
        set. Transitive closure algorithm is applied.

        """
        ns = set(G.nonterminals)

        NULLABLE = {X for X, rhs in G.rules if not rhs}     # :: Set[<terminal>]
        def nullable(X, path=()):
            if X in ns and X not in path:
                for _, rhs in G[X]:
                    if rhs:
                        path += (X,)
                        if all(nullable(Y, path) for Y in rhs):
                            NULLABLE.add(X)
        for n in ns:
            nullable(n)

        FIRST = {}           # :: Dict[<nonterminal>: Set[<terminal>]]
        def first1(X, path=()):
            if X in path:
                return {}
            elif X not in ns:
                return {X}
            else:
                F = set()
                if X in NULLABLE:
                    F.add(EPSILON)
                for _, rhs in G[X]:
                    for Y in rhs:
                        path += (X,)
                        F.update(first1(Y, path))
                        if Y not in NULLABLE:
                            break
                if X not in FIRST:
                    FIRST[X] = F
                else:
                    FIRST[X].update(F)
                return F

        for n in ns:
            first1(n)

        G.NULLABLE = NULLABLE
        G.FIRST = FIRST
        G.first1 = first1

        def first_of_seq(seq, tail=DUMMY):
            fs = set()
            for X in seq:
                fs.update(G.first1(X))
                if EPSILON not in fs:
                    return fs
            fs.discard(EPSILON)
            # Note :tail: can also be EPSILON
            fs.add(tail)
            return fs

        G.first_of_seq = first_of_seq

    def tokenize(G, inp: str, with_end: False):
        """Perform lexical analysis
        with given string. Yield feasible matches with defined lexical patterns.
        Ambiguity is resolved by the order of patterns within definition.

        It must be reported if `pos` is not strictly increasing when
        any of the valid lexical pattern matches zero-length string!!
        This may lead to non-termination.
        """

        pos = 0
        while pos < len(inp):
            for lex, rgx in G.terminals.items():
                m = rgx.match(inp, pos=pos)
                if m and len(m.group()) > 0: # Guard matching for zero-length.
                    break
            if m and lex == IGNORED:
                at, pos = m.span()
            elif m and lex != IGNORED:
                at, pos = m.span()
                yield Token(at, lex, m.group())
            else:
                # Report unrecognized Token here!
                at = pos
                pos += 1
                yield Token(at, ERROR, inp[at])
        if with_end:
            yield Token(pos, END, END_PATTERN_DEFAULT)

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
                for j, jrl in enumerate(G.rules):
                    if itm.actor == jrl.lhs:
                        jtm = G.make_item(j, 0)
                        if jtm not in C:
                            C.append(jtm)
            z += 1
        return C

    def closure1_with_lookahead(G, item, a):
        """Fig 4.40 in Dragon Book.

        CLOSURE(I)
            J = I.copy()
            for (A -> α.Bβ, a) in J:
                for (B -> γ) in G:
                    for b in FIRST(βa):
                        if (B -> γ, b) not in J:
                            J.add((B -> γ, b))
            return J


        This can be done before calculating LALR-Item-Sets, thus avoid
        computing closures repeatedly by applying the virtual DUMMY
        lookahead(`#` in the dragonbook). Since this lookahead must
        not be shared by any symbols within any instance of Grammar, a
        special value is used as the dummy(Not including None, since
        None is already used as epsilon in FIRST set).

        For similar implementations within lower-level language like
        C, this value can be replaced by any special number which
        would never represent a unicode character.

        """
        C = [(item, a)]
        z = 0
        while z < len(C):
            itm, a = C[z]
            if not itm.ended():
                for j, jrl in enumerate(G.rules):
                    if itm.actor == jrl.lhs:
                        # Dreprecated way of
                        # beta = itm.look_over
                        # jlk = []
                        # for X in beta + [a]:
                        #     for b in G.first1(X):
                        #         if b is not EPSILON and b not in jlk:
                        #             jlk.append(b)
                        #     if EPSILON not in G.first1(X):
                        #         break
                        # for b in jlk:
                        #     jtm = G.make_item(j, 0)
                        #     if (jtm, b) not in C:
                        #         C.append((jtm, b))
                        for b in G.first_of_seq(itm.look_over, a):
                            jtm = G.make_item(j, 0)
                            if (jtm, b) not in C:
                                C.append((jtm, b))
            z += 1
        return C


class assoclist(list):

    """This class intends to be cooperated with metaclass definition
    through __prepare__ method. When returned in __prepare__, it will
    be used by registering class-level method definitions. Since it
    overrides the setter and getter of default `dict` supporting class
    definition, it allows repeated method declaration to be registered
    sequentially in a list. As the result of such class definition,
    declarated stuff can be then extracted in the __new__ method in
    metaclass definition.

    Assoclist :: [(k, v)]

    """

    def __setitem__(self, k, v):
        ls = super(assoclist, self)
        ls.append((k, v))

    def __getitem__(self, k0):
        vs = [v for k, v in self if k == k0]
        if vs:
            return vs[0]
        else:
            raise KeyError('No such attribute.')


class cfg(type):

    """This metaclass performs accumulation for all declared grammar
    lexical and syntactical rules at class level, where declared
    variables are registered as lexical elements and methods are
    registered as Rule objects defined above.

    """

    @classmethod
    def __prepare__(mcls, n, bs, **kw):
        return assoclist()

    def __new__(mcls, n, bs, accu):
        """After gathering definition of grammar elements, reorganization
        and first checkings are done in this method, resulting in creating
        a Grammar object.

        """
        lexes = OrderedDict()
        rules = []
        attrs = []

        for k, v in accu:
            # Handle lexical declaration.
            if not k.startswith('_') and isinstance(v, str):
                if k in lexes:
                    raise GrammarError('Repeated declaration of lexical symbol {}.'.format(k))
                lexes[k] = v
            # Handle implicit rule declaration through methods.
            elif not k.startswith('_') and callable(v):
                rules.append(Rule(v))
            # Handle explicit rule declaration through decorated methods.
            elif isinstance(v, Rule):
                rules.append(v)
            # Handle normal private attributes/methods.
            # These must be prefixed with (at least one) underscore
            else:
                attrs.append((k, v))

        # Default matching order of special patterns:
        # - Always match END first;
        # - Always match IGNORED secondly, if it is not specified;
        # - Always match ERROR at last;

        if IGNORED not in lexes:
            lexes[IGNORED] = IGNORED_PATTERN_DEFAULT
            lexes.move_to_end(IGNORED, last=False)

        # END pattern is not overridable.
        lexes[END] = END_PATTERN_DEFAULT
        lexes.move_to_end(END, last=False)

        # ERROR pattern has the lowest priority, meaning it is only
        # matched after failing matching all other patterns. It may be
        # overriden by the user.
        if ERROR not in lexes:
            lexes[ERROR] = ERROR_PATTERN_DEFAULT

        return Grammar(lexes, rules, attrs)


"""
The above parts are utitlies for grammar definition and extraction methods
of grammar information.

The following parts supplies utilities for parsing.
"""

# The object ParseLeaf is semantically indentical to the object Token.
ParseLeaf = Token


class ParseTree(object):

    def __init__(self, rule: Rule, subs: list):
        # Contracts
        assert isinstance(rule, Rule)
        for sub in subs:
            assert isinstance(sub, ParseTree) or isinstance(sub, ParseLeaf)
        self.rule = rule
        self.subs = subs

    """Postorder tree traversal with double stacks scheme.
    Example:
    - prediction stack
    - processed argument stack

    i-  [T]$
    a- $[]

    =(T -> AB)=>>

    i-  [A B (->T)]$
    a- $[]

    =(A -> a1 a2)=>>

    i-  [a1 a2 (->A) B (->T)]$
    a- $[]

    =a1, a2=>>

    i-  [(->A) B (->T)]$
    a- $[a1 a2]

    =(a1 a2 ->A)=>>

    i-  [B (->T)]$
    a- $[A]

    =(B -> b)=>>

    i-  [b (->B) (->T)]$
    a- $[A]

    =b=>>

    i-  [(->B) (->T)]$
    a- $[A b]

    =(b ->B)=>>

    i-  [(->T)]$
    a- $[A B]

    =(A B ->T)=>>

    i-  []$
    a- $[T]

    """

    def translate(tree, trans_tree=None, trans_leaf=None):
        # Direct translation
        if not trans_tree:
            trans_tree = lambda t, args: t.rule.seman(*args)
        if not trans_leaf:
            trans_leaf = lambda tok: tok.value
        # Aliasing push/pop, simulating stack behavior
        push, pop = list.append, list.pop
        # Signal for prediction/reduce
        P, R = 0, 1
        sstk = [(P, tree)]
        astk = []
        while sstk:
            i, t = pop(sstk)
            if i == R:
                args = []
                for _ in t.subs:
                    push(args, pop(astk))
                # apply semantics
                push(astk, trans_tree(t, args[::-1]))
            elif isinstance(t, ParseLeaf):
                push(astk, trans_leaf(t))
            elif isinstance(t, ParseTree):
                # mark reduction
                push(sstk, (R, t))
                for sub in reversed(t.subs):
                    push(sstk, (P, sub))
            else:
                assert False, (i, t)
        assert len(astk) == 1
        return astk.pop()

    def to_tuple(self):
        """Translate the parse tree into Python tuple form. """
        return self.translate(lambda t, subs: (t.rule.lhs, subs),
                              lambda tok: tok)

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()

    def __repr__(self):
        """Use pformat to handle structural hierarchy."""
        return pp.pformat(self.to_tuple())


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


class ParserBase(object):

    """Abstract class for parsers. """

    def __init__(self, grammar):
        self.grammar = grammar
        for k, v in grammar.attrs:
            assert k.startswith('_')
            setattr(self, k, v)

    def __repr__(self):
        # Polymorphic representation without overriding
        # raise NotImplementedError('Parser should override __repr__ method. ')
        return self.__class__.__name__ + 'Parser-{}'.format(self.grammar)

    def parse(self, inp: str):
        # Must be overriden
        raise NotImplementedError('Any parse should have a `parse` method.')


@meta
class Earley(ParserBase):

    """Earley parser is able to parse ANY Context-Free Language properly
    using the technique of dynamic programming. It performs
    non-deterministic parsing since all parse results due to potential
    ambiguity of given grammar should be found.

    The underlying parse forest is indeed a Tomita-style *Graph
    Structured Stack*. But such structure cannot be directly traversed
    for the conventional tree-based parsing semantics. In order to
    prepare valid traverses, the result here is a list of parse trees
    where some subtrees are replicatedly referenced by these trees.

    However the parsing process is relatively slow due to the
    on-the-fly computation of item set closures, as well as the
    subtree stacks prepared for completing parse trees.

    """

    def __init__(self, grammar):
        super(Earley, self).__init__(grammar)

    def recognize(self, inp: str):
        """Naive Earley's recognization algorithm. No parse produced.

        """
        G = self.grammar
        S = [[(0, G.make_item(0, 0))]]

        for k, tok in enumerate(G.tokenize(inp, with_end=True)):

            C = S[-1]
            C1 = []

            z_j = 0
            while z_j < len(C):
                j, jtm = C[z_j]
                # ISSUES by grammars having ε-Rules:
                #
                # - In grammar without nullable rules, each predicted
                #   non-kernel item consumes no further nonterminal
                #   symbols, which means each completion item has no
                #   effect on generated prediction items and can be
                #   processed only ONCE and merely SINGLE-PASS-CLOSURE
                #   is needed;
                #
                # - BUT in grammar with nullable rules, each predicted
                #   non-kernel item can consume arbitrarily many
                #   further nonterminals which leads to its
                #   completion, meaning prediction may result in new
                #   completion items, leading to repeated processing
                #   of the identical completion item. Thus
                #   LOOP-closure is needed.
                #
                if not jtm.ended():
                    # Prediction/Completion no more needed when
                    # recognizing END-Token
                    if not tok.is_END():
                        # Prediction: find nonkernels
                        if jtm.actor in G.nonterminals:
                                for r, rule in enumerate(G.rules):
                                    if rule.lhs == jtm.actor:
                                        ktm = G.make_item(r, pos=0)
                                        new = (k, ktm)
                                        if new not in C:
                                            C.append(new)
                        # Scanning/proceed for next States
                        elif jtm.actor == tok.symbol:
                            C1.append((j, jtm.shifted))
                else: # jtm.ended() at k
                    for (i, itm) in S[j]:
                        if not itm.ended() and itm.actor == jtm.target:
                            new = (i, itm.shifted)
                            if new not in C:
                                C.append(new)
                z_j += 1
            if not C1:
                if not tok.is_END():
                    msg = '\n####################'
                    msg += '\nUnrecognized {}'.format(tok)
                    msg += '\nChoked active ItemSet: \n{}\n'.format(pp.pformat(C))
                    msg += '####################\n'
                    msg += '\nStates:'
                    msg += '\n{}\n'.format(pp.pformat(S))
                    raise ParserError(msg)
            else:
                # Commit proceeded items (by Scanning) as item set.
                S.append(C1)

        return S

    def parse_chart(self, inp: str):
        """Perform chart-parsing framework method with Earley's recognition
        algorithm. The result is a chart, which semantically includes
        Graph Structured Stack i.e. all possible parse trees.

        Semantics of local variables:

        :tokens:

            Caching the input tokens;

        :chart:

            The graph recording all recognitions, where chart[j][k] is
            a set containing all recognized parsings on input segment
            from j'th to k'th tokens.

        :agenda:

            During the Earley parsing process, the agenda caches all
            items that should be processed when consuming the k'th
            input token.

        """
        G = self.grammar
        # chart[i][j] registers proceeding items from @i to @j
        chart = self.chart = defaultdict(lambda: defaultdict(set))
        agenda = [(0, G.make_item(0, 0))]

        for k, tok in enumerate(G.tokenize(inp, False)):
            agenda1 = []
            while agenda:
                j, jtm = agenda.pop()
                # Directly register intermediate state item
                if jtm not in chart[j][k]:
                    chart[j][k].add(jtm)
                if not jtm.ended():
                    if not tok.is_END():
                        # Prediction
                        if jtm.actor in G.nonterminals:
                            for r, rule in enumerate(G.rules):
                                if rule.lhs == jtm.actor:
                                    ktm = G.make_item(r, 0)
                                    if ktm not in chart[k][k]:
                                        chart[k][k].add(ktm)
                                        agenda.append((k, ktm))
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
            else:
                raise ParserError('Fail: empty new agenda by {}\nchart:\n{}'.format(tok,
                    pp.pformat(chart)))

        return chart

    def parse_forest(self, inp: str):
        """Construct single-threaded parse trees (the Parse Forest based upon
        the underlying Graph Structured Stack and from another
        perspective, the chart) during computing Earley item sets.

        The most significant augmentation w.R.t. the naive recognition
        algorithm is when more than one completed items @jtm in the
        agenda matches one parent @(itm)'s need, the parent items'
        stacks need to be forked/copied to accept these different @jtm
        subtrees as a feasible shift.

        Since parse trees are computed on-the-fly, the result is a set
        of feasible parse trees.

        CAUTION: Interpretation on-the-fly is NOT allowed! Since the
        traversal execution order of sibling subtrees' semantic
        behaviors should be preserved for their parent tree, whereas
        during AGENDA completion the execution of some shared item's
        behavior may be interleaving (rigor proof still absent). OTOS,
        after all top trees have been constructed, the traverses by
        each can deliver correct semantic results.

        """
        G = self.grammar
        # state :: {<Item>: [<Stack>]}
        s0 = {(0, G.make_item(0, 0)): [()]}
        ss = self.forest = []
        ss.append(s0)

        for k, tok in enumerate(G.tokenize(inp, with_end=True)):

            # Components for the behavior of One-Pass transitive closure
            # - The accumulated set as final closure
            s_acc = ss[-1]
            # - The set of active items to be processed, i.e. AGENDA
            s_act = {**s_acc}
            # - The initial set for next token position
            s_after = {}

            while 1:
                # - The augmention items predicted/completed by items
                #   in the AGENDA @s_act
                s_aug = {}

                for (j, jtm), j_stks in s_act.items():
                    # prediction
                    # TODO: Support nullable grammar (cancel LOOP) by applying
                    # predictor which
                    # - can predict with overlooking sequential nullable symbols
                    # - can conclude completion of nullable symbols
                    if not jtm.ended():
                        if tok.symbol != END:
                            if jtm.actor in G.nonterminals:
                                for r, rule in enumerate(G.rules):
                                    if rule.lhs == jtm.actor:
                                        new = (k, G.make_item(r, 0))
                                        if new not in s_acc:
                                            s_aug[new] = [()]
                            # scanning
                            elif jtm.actor == tok.symbol:
                                if (j, jtm.shifted) not in s_after:
                                    s_after[j, jtm.shifted] = []
                                for j_stk in j_stks:
                                    s_after[j, jtm.shifted].append(j_stk + (tok,))
                    # completion
                    else:
                        for j_stk in j_stks:
                            j_tr = ParseTree(jtm.rule, j_stk)
                            for (i, itm), i_stks in ss[j].items():
                                if not itm.ended() and itm.actor == jtm.target:
                                    new = (i, itm.shifted)
                                    if new not in s_aug:
                                        s_aug[new] = []
                                    for i_stk in i_stks:
                                        s_aug[new].append(i_stk + (j_tr,))
                if s_aug:
                    # Register new AGENDA items as well as their
                    # corresponding completion stacks
                    for (i, itm), i_stk in s_aug.items():
                        if (i, itm) in s_acc:
                            s_acc[i, itm].extend(i_stk)
                        else:
                            s_acc[i, itm] = i_stk
                    # Next pass of AGENDA
                    s_act = s_aug
                else:
                    # No new AGENDA items
                    break

            if tok.symbol != END:
                ss.append(s_after)

        return ss

    def parse(self, inp: str, interp=False):
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
                    tree = ParseTree(G.top_rule, i_stk)
                    if interp:
                        fin.append(tree.translate())
                    else:
                        fin.append(tree)
        return fin

    def interpret(self, inp: str):
        return self.parse(inp, True)


@meta
class GLR(ParserBase):

    """GLR parser is a type of non-deterministic parsers which produces
    parse forest rather than exactly one parse tree, like the Earley
    parser. Similarly, The traversal order of semantic behaviors of
    subtrees under the same parent rule should be preserved until full
    parse is generated. This means execution of semantic behavior
    during parsing process is banned.

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

    def __init__(self, grammar):
        super(GLR, self).__init__(grammar)
        self._build_automaton()

    def _build_automaton(self):

        """Calculate general LR(0)-Item-Sets with no respect to look-aheads.
        Each conflict is registered into the parsing table. For
        practical purposes, these conflicts should be reported for the
        grammar writer to survey the conflicts and experiment with
        potential ambiguity, thus achieving better inspection into the
        characteristics of the grammar itself.

        For LR(0) grammars, the performance of GLR is no significantly
        worse than the LALR(1) parser.

        """

        G = self.grammar

        # Kernels
        self.Ks = Ks = [[G.make_item(0, 0)]]
        self.GOTO = GOTO = []
        self.ACTION = ACTION = []

        # Construct LR(0)-DFA
        k = 0
        while k < len(Ks):

            I = Ks[k]
            iacts = {'reduce': [], 'shift': {}}
            igotoset = OrderedDict()

            for itm in G.closure(I):
                if itm.ended():
                    iacts['reduce'].append(itm)
                else:
                    X = itm.actor
                    jtm = itm.shifted
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

            ACTION.append(iacts)
            GOTO.append(igoto)

            k += 1

    def parse(self, inp: str):

        """Note during the algorithm, When forking the stack, there may be
        some issues:

        - SHIFT consumes a token, while REDUCE consumes no.

        - The Overall-Stack-Set can be maintained as a
          stack/queue. For each stack element in the
          Overall-Stack-Set, keep a position index in the input token
          sequence (or a cached stream iterator) associated with each
          stack element. This allows virtual backtracking.

            * As a result, the probes for each stack in
              Overall-Stack-Set can be done in a DFS/BFS/Best-FS
              manner.

            * In case no lookahead information is incorperated, the
              GLR parser can keep track of all viable Partial Parsing
              all along the process.

            * Panic-mode error ignorance does not fit good for GLR
              parser, since ANY sub-sequence of the ordered input
              tokens can be used to construct a parse tree. A overly
              large amount of "partially correct" parse trees may be
              delivered.

        """

        G = self.grammar
        GOTO = self.GOTO
        ACTION = self.ACTION

        results = []
        tokens = list(G.tokenize(inp, False))
        forest = [([0], [], 0)]          # forest :: [[State-Number, [Tree], InputPointer]]

        while forest:

            # DFS/BFS depends on the pop direction
            stk, trs, i = forest.pop()

            # State on the stack top
            st = stk[-1]
            reds = ACTION[st]['reduce']
            shif = ACTION[st]['shift']

            # REDUCE
            # There may be multiple reduction options. Each option leads
            # to ONE NEW FORK of the parsing thread.
            for ritm in reds:
                # Forking, copying State-Stack and trs
                # Index of input remains unchanged.
                frk = stk[:]
                trs1 = trs[:]
                subts = []
                for _ in range(ritm.size):
                    frk.pop()   # only serves amount
                    subts.insert(0, trs1.pop())
                trs1.append(ParseTree(ritm.rule, subts))

                # Deliver/Transit after reduction
                if ritm.target == G.top_rule.lhs:
                    # Discard partial top-tree, which can be legally
                    # completed while i < len(tokens)
                    if i == len(tokens):
                        results.append(trs1[0])
                else:
                    frk.append(GOTO[frk[-1]][ritm.target])
                    forest.append([frk, trs1, i]) # index i stays

            # SHIFT
            # There can be only 1 option for shifting given a symbol due
            # to the nature of LR automaton.
            if i < len(tokens):
                tok = tokens[i]
                if tok.symbol in shif:
                    stk.append(GOTO[st][tok.symbol])
                    trs.append(tok)
                    forest.append([stk, trs, i+1]) # index i increases

                # ERROR
                elif not reds:
                    # Under panic mode, some unwanted prediction rules
                    # may be proceeded till the end while discarding
                    # arbitrarily many tokens. In other words, every
                    # substring of the input token sequence might be
                    # used to make a parse. This comprises another
                    # problem: Finding the "optimal" parse which
                    # ignores least amount of tokens, or least
                    # significant set of tokens w.R.t. some criterien.
                    #
                    # msg = '\n'.join([
                    #     '',
                    #     '#########################',
                    #     'GLR - Ignoring syntax error by Token {}'.format(tok),
                    #     ' Current left-derivation fork:\n{}'.format(
                    #         pp.pformat([self.Ks[i] for i in stk])),
                    #     '#########################',
                    #     '',
                    # ])
                    # warnings.warn(msg)
                    # Push back
                    # forest.append([stk, trs, i+1])
                    pass

        return results

    def interpret(self, inp: str):
        res = self.parse(inp)
        return [tree.translate() for tree in res]


@meta
class LALR(ParserBase):

    """Lookahead Leftmost-reading Rightmost-derivation (LALR) parser may
    be the most widely used parser type. It is a parser generator which
    pre-computes automaton before any parsing work is executed.

    Due to its deterministic nature and table-driven process does it
    have linear-time performance.

    """

    def __init__(self, grammar):
        super(LALR, self).__init__(grammar)
        self._build_automaton()

    def _build_automaton(self):

        G = self.grammar

        Ks = [[G.make_item(0, 0)]]   # Kernels
        goto = []

        # Calculate Item Sets, GOTO and propagation graph in one pass.
        i = 0
        while i < len(Ks):

            K = Ks[i]

            # Use OrderedDict to preserve order of finding
            # goto's, which should be the same order with
            # the example in textbook.
            igoto = OrderedDict()

            # SetOfItems algorithm
            for itm in G.closure(K):
                # If item (A -> α.Xβ) has a goto.
                if not itm.ended():
                    X = itm.actor
                    jtm = itm.shifted
                    if X not in igoto:
                        igoto[X] = []
                    if jtm not in igoto[X]:
                        igoto[X].append(jtm)

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

            i += 1

        # The table `spont` represents the spontaneous lookaheads at
        # first. But it can be used for in-place updating of
        # propagated lookaheads. After the whole propagation process,
        # `spont` is the final lookahead table.
        spont = [OrderedDict((itm, set()) for itm in K)
                 for K in Ks]

        # Initialize spontaneous END token for the top item set.
        init_item = Ks[0][0]
        spont[0][init_item].add(END)
        for ctm, a in G.closure1_with_lookahead(init_item, END):
            if not ctm.ended():
                X = ctm.actor
                j0 = goto[0][X]
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
                        j = goto[i][X]
                        if a != DUMMY:
                            spont[j][ctm.shifted].add(a)
                        else:
                            # Propagation from KERNEL to its target
                            # derived by the KERNEL's closure. See
                            # algo 4.62 in Dragon book.
                            propa[i].append((ktm, j, ctm.shifted))

        b = 1
        while True:
            brk = True
            for i, _ in enumerate(Ks):
                for itm, j, jtm in propa[i]:
                    lks_src = spont[i][itm]
                    lks_tar = spont[j][jtm]
                    for a in lks_src:
                        if a not in lks_tar:
                            lks_tar.add(a)
                            brk = False
            if brk:
                G.passes = b
                break
            else:
                b += 1

        self.Ks = Ks
        self.GOTO = goto
        self.propa = propa
        self.table = table = spont

        # spont has included all the non-kernel items which are not
        # necessary if spont registers only the target, not the
        # source.  A table representation covering only kernel items
        # to preserve for simplicity.
        self.ktable = []
        for i, K in enumerate(Ks):
            klk = {}
            for k in K:
                klk[k] = spont[i][k]
            self.ktable.append(klk)

        # Construct ACTION table
        ACTION = [{} for _ in table]

        # SHIFT for non-ended to consume terminal for transition.
        for i, xto in enumerate(goto):
            for a, j in xto.items():
                if a in G.terminals:
                    ACTION[i][a] = ('shift', j)

        # REDUCE for ended to reduce.
        conflicts = []
        for i, itm_lks in enumerate(table):
            for itm, lks in itm_lks.items():
                for lk in lks:
                    for ctm, lk1 in G.closure1_with_lookahead(itm, lk):
                        if ctm.ended():
                            if lk1 in ACTION[i] and ACTION[i][lk1] != ('reduce', ctm):
                                conflicts.append((i, lk1, ctm))
                            else:
                                ACTION[i][lk1] = ('reduce', ctm)
                # # Accept-Item
                if itm.index_pair == (0, 1):
                    ACTION[i][END] = ('accept', None)
        if conflicts:
            msg = ''
            for i, lk, itm in conflicts:
                msg += '\n'.join([
                    '\n! LALR-Conflict raised:',
                    '  - in ACTION[{}]: '.format(i),
                    '{}'.format(pp.pformat(ACTION[i])),
                    "  * conflicting action on token {}: ".format(repr(lk)),
                    "{{{}: ('reduce', {})}}".format(repr(lk), itm)
                ])
            msg = '\n########## Error ##########\n {} \n#########################\n'.format(msg)
            raise ParserError(msg)

        self.ACTION = ACTION

    def parse(self, inp: str, interp=False):

        """Perform table-driven deterministic parsing process. Only one parse
        tree is to be constructed.

        If `interp` mode is turned True, then a parse tree is reduced
        to semantic result once its sub-nodes are completed, otherwise
        the parse tree is returned.

        """

        # Aliasing
        trees = []
        sstack = self.sstack = [0]
        G = self.grammar
        GOTO = self.GOTO
        ACTION = self.ACTION

        toker = self.grammar.tokenize(inp, with_end=True) # Use END to force finishing by ACCEPT
        tok = next(toker)

        try:
            while 1:

                i = sstack[-1]

                if tok.symbol not in ACTION[i]:
                    msg = '\n'.join([
                        '',
                        '#########################',
                        'LALR - Ignoring syntax error by Token {}'.format(tok),
                        ' Current left-derivation stack:\n{}'.format(
                            pp.pformat([self.Ks[i] for i in sstack])),
                        '#########################',
                        '',
                    ])
                    warnings.warn(msg)
                    tok = next(toker)

                else:

                    act, arg = ACTION[i][tok.symbol]

                    # SHIFT
                    if act == 'shift':
                        if interp:
                            trees.append(tok.value)
                        else:
                            trees.append(tok)
                        sstack.append(GOTO[i][tok.symbol])
                        # Go on iteration/scanning
                        tok = next(toker)

                    # REDUCE
                    elif act == 'reduce':
                        rtm = arg
                        ntar = rtm.target
                        subts = []
                        for _ in range(rtm.size):
                            subt = trees.pop()
                            subts.insert(0, subt)
                            sstack.pop()
                        if interp:
                            tree = rtm.eval(*subts)
                        else:
                            tree = ParseTree(rtm.rule, subts)
                        trees.append(tree)
                        # New got symbol is used for shifting.
                        sstack.append(GOTO[sstack[-1]][ntar])

                    # ACCEPT
                    elif act == 'accept':
                        return trees[-1]

                    else:
                        raise ParserError('Invalid action {} on {}'.format(act, arg))

        except StopIteration:
            raise ParserError('No enough tokens for completing the parse. ')

    def interpret(self, inp):
        return self.parse(inp, interp=True)


@meta
class WLL1(ParserBase):
    """Weak-LL(1)-Parser.

    Since strong-LL(1) grammar parser includes the usage of FOLLOW
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
    recognition error in compared with strong-LL(1) parser.

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
                for a in G.first1(rhs[0]):
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

    def parse(self, inp: str, interp=False):
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
        except StopIteration as e:
            raise ParserError('No enough token for complete parsing.')


@meta
class GLL1(ParserBase):
    """Generalized LL(1) Parser. """

    def __init__(self, grammar):
        super(GLL1, self).__init__(grammar)
        self._calc_gll1_table()

    def _calc_gll1_table(self):
        G = self.grammar
        table = self.table = {}
        for r, rule in enumerate(G.rules):
            lhs, rhs = rule
            if lhs not in table:
                table[lhs] = defaultdict(list)
            if rhs:
                for a in G.first_of_seq(rhs, EPSILON):
                    if a is not EPSILON:
                        table[lhs][a].append(rule)

    def parse(self, inp: str):
        """Table-driven GLL1 parsing based on BFS (unlike typical LL1 with
        backtracking implementing full DFS).

        Tracing parallel stacks, where signal (α>A) means reducing
        production A with sufficient amount of arguments on the
        argument stack.

        tokens:  | aβd#
        threads: | (#      ,      S#) :: ({arg-stk}, {pred-stk})

        ==(S -> aBc), (S -> aD)==>>

        | aβd#
        | (#      ,    aBd(aBd>S)#)
          (#      ,    aDe(aD>S)#)

        ==a==>>

        | βd#
        | (#a     ,     Bd(aBd>S)#)
          (#a     ,     De(aD>S)#)

        ==prediction (B -> b h)
          prediction (D -> b m), push predicted elements into states ==>>

        | βd#
        | (#a     ,bh(bh>B)d(aBd>S)#)
          (#a     ,bm(bm>D)De(aD>S)#)

        ==b==>

        | βd#
        | (#ab    ,h(bh>B)d(aBd>S)#)
          (#ab    ,m(bm>D)De(aD>S)#)

        ==h==>

        | βd#
        | (#abh   ,(bh>B)d(aBd>S)#)
          (#ab    ,{FAIL} m(bm>D)De(aD>S)#)

        ==(reduce B -> β), push result into args ==>>

        | βd#
        | (#aB    ,d(aBd>S)#)

        and further.

        """
        push, pop = list.append, list.pop
        table = self.table
        G = self.grammar
        # 
        PRED, REDU = 0, 1
        threads = [([], [(PRED, G.top_symbol)])] 
        # 
        for k, tok in enumerate(G.tokenize(inp)):
            at, look, lexeme = tok
            acc = threads[k]
            agenda = threads[k][:]
            while 1:
                agenda1 = []
                for (astk, pstk) in agenda:
                    if not pstk:
                        # Deliver result?
                        yield astk[-1]
                    else:
                        act, actor = pstk.pop(0)
                        if act is PRED:
                            if actor in G.nonterminals:
                                if look in table[actor]:
                                    for rule in table[actor][look]:
                                        nps = [(PRED, x) for x in rule.rhs]
                                        agenda1.append(
                                            (astk[:], [*nps, (REDU, rule.lhs), *pstk]))
                            else:
                                if look == actor:
                                    if interp: astk.append(lexeme)
                                    else: astk.append(tok)
                                    agenda1.append((astk, pstk))
                        else:
                            subs = []
                            for _ in actor.rhs:
                                subs.insert(0, astk.pop())
                            t = ParseTree(actor.rhs, subs)
                            if interp: astk.append(t.translate())
                            else: astk.append(t)
                            agenda1.append((astk, pstk))
                if not agenda1:
                    break
                else: 
                    acc.extend(agenda1)
                    agenda = agenda1

# class S(metaclass=lalr):

#     a = r'a'

#     @Rule
#     def S(a, S):
#         return a + S
#     @Rule
#     def S(a):
#         return 'A'

# pp.pprint(S)
# pp.pprint(S.interpret('a  a   a  a'))

# S -> a A b
#    | b A a
# A -> c S
#    | ε
#
# {a, b, c}


# class fll1(cfg):
#     def __new__(mcls, n, bs, kw):
#         grammar = cfg.__new__(mcls, n, bs, kw)
#         return WLL1(grammar)


# class SExp(metaclass=WLL1.meta):
#     l1 = r'\('
#     r1 = r'\)'
#     symbol = r'[^\(\)\s]+'
#     def SExp(symbol):
#         return symbol
#     def SExp(l1, SExps, r1):
#         return SExps
#     def SExps():
#         return []
#     def SExps(SExp, SExps):
#         return [SExp] + SExps


# # pp.pprint(SExp.table)
# res = SExp.parse('( ( (a ab  xyz  cdef) just ))', interp=0)
# pp.pprint(res)
# # pp.pprint(res.to_tuple())
# pp.pprint(res.translate())
# # pp.pprint(SExp.parse('( ( (a ab  xyz  cdef) just ))', interp=1))
