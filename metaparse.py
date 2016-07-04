"""This module prepares an alternative utilities for syntactical
analyzing in pure Python environment. It includes:

- OO-style grammar object system for context-free languages with
  completeness check

- Elegant grammar definer based upon metaprogramming

- Parser interface

- Optional parsing algorithms
  - LL(1)
  - Earley
  - LALR

To allow the ultimate ease of usage, the OO-style grammar definition
is approached by the Python class structure, by treating the method
definitions as syntactic rules associated with semantic behaviors.

Although such representation of grammar is somewhat verbose especially
by the definition of alternative productions, it clearly specifies
alternative semantic behaviors with NAMED parameters, which appears to
be more expressive than POSITIONAL parameters.

"""

import re
import inspect
import pprint as pp

from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple
from collections import deque

class stack(list):
    push = list.append


class GrammarError(Exception):
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


class _EPSILON:
    """A singleton object representing empty productivity of nonterminals."""

    def __repr__(self):
        return 'EPSILON'


EPSILON = _EPSILON()

# Object Token
Token = namedtuple('Token', 'at symbol value')
Token.start = property(lambda s: s.at)
Token.end = property(lambda s: s.at + len(s.value))
Token.__repr__ = lambda s: '({} -> {})@[{}:{}]'.format(s.symbol, repr(s.value), s.at, s.end)

# class Token(object):
#
#     def __init__(self, at, symbol, value):
#         self.at = at
#         self.symbol = symbol
#         self.value = value
#
#     def __repr__(self):
#         return '({} -> {})@[{}:{}]'.format(
#             self.symbol,
#             repr(self.value),
#             self.at,
#             self.at + len(self.value))
#
#     def __eq__(self, other):
#         return self.symbol == other.symbol
#
#     def __iter__(self):
#         yield self.at
#         yield self.symbol
#         yield self.value
#
#     @property
#     def start(self):
#         return self.at
#
#     @property
#     def end(self):
#         return self.at + len(self.value)



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
            # Tailing numeric subscript like xxx_4
            s = re.search(r'_(\d+)$', x)
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

    def over_rest(s):
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

        lexes  :: odict{str<terminal-name> : str<terminal-pattern>}
            - A ordered list of pairs representing lexical rules.
        rules  :: [Rule]
            - A list of grammar rules.

        Notes:

        - Checks whether there is a singleton TOP-rule
          and add such one if not.

        - Needs to perform validity and well-formity!!
            - Undeclared tokens;
            - Undeclared nonterminals;
            - Unused tokens;
            - Unreachable nonterminals/rules;
            - Cyclic rules;

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
            raise ValueError(msg)
        for t in _t_unused:
            print('Warning: Unused terminal symbol {}'.format(t))
        for nt in _nt_unused:
            print('Warning: Unused nonterminal symbol {}'.format(nt))


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

        # Register other attributes
        for k, v in attrs:
            assert k.startswith('_')
            setattr(self, k, v)

        self.rules = rules
        self.symbols = self.nonterminals + [a for a in lexes if a != END]

        # Top rule of Augmented Grammar.
        self.start_rule = self.rules[0]
        self.start_symbol = self.start_rule.lhs

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
        """Create a pair of integers indexing the rule and
        active position.
        """
        return Item(G.rules, r, pos)

    def _calc_first_and_nullable(G):
        """Calculate the FIRST set of this grammar as well
        as the NULLABLE set. Transitive closure algorithm
        is applied.
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
                # Report error here!
                # print('Unrecognized token at {} with value {}!'.format(at, m.group()))
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

    def closure_with_lookahead(G, item, a):
        """
        Fig 4.40 in Dragon Book.

        CLOSURE(I)
            I = I.copy()
            for (A -> α.Bβ, a) in I:
                for (B -> γ) in G:
                    for b in FIRST(βa):
                        if (B -> γ, b) not in I:
                            I.add((B -> γ, b))
            return I


        This can be done before calculating LALR-Item-Sets, thus avoid
        computing closures repeatedly by applying the virtual dummy
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
                        jlk = []
                        beta = itm.over_rest()
                        for X in beta + [a]:
                            for b in G.first1(X):
                                if b is not EPSILON and b not in jlk:
                                    jlk.append(b)
                            if EPSILON not in G.first1(X):
                                break
                        for b in jlk:
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
                    raise ValueError('Repeated declaration of lexical symbol {}.'.format(k))
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

        # May be overwritten.
        # Consider default matching order for these Lexers:
        # - Always match END first;
        # - Always match IGNORED secondly;
        # - Always match ERROR at last;

        if IGNORED not in lexes:
            lexes[IGNORED] = IGNORED_PATTERN_DEFAULT
            lexes.move_to_end(IGNORED, last=False)

        lexes[END] = END_PATTERN_DEFAULT
        lexes.move_to_end(END, last=False)

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
        self.rule = rule
        self.subs = subs

    """Postorder tree traversal with double stacks scheme.
    Example:
    - input stack
    - args stack

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
    a- $[a1! a2!]

    =(a1! a2! ->A)=>>

    i-  [B (->T)]$
    a- $[A!]

    =(B -> b)=>>

    i-  [b (->B) (->T)]$
    a- $[A!]

    =b=>>

    i-  [(->B) (->T)]$
    a- $[A! b!]

    =(b! ->B)=>>

    i-  [(->T)]$
    a- $[A! B!]

    =(A! B! ->T)=>>

    i-  []$
    a- $[T!]

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
                # raise ValueError('Invalid state stack content.')
        assert len(astk) == 1
        return astk.pop()

    def to_tuple(self):
        """Translate the parse tree into Python tuple form.

        """
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

    def __repr__(self):
        # Polymorphic representation without overriding
        # raise NotImplementedError('Parser should override __repr__ method. ')
        return self.__class__.__name__ + 'Parser-{}'.format(self.grammar)

    def parse(self):
        # Must be overriden
        raise NotImplementedError('Any parse should have a `parse` method.')


@meta
class Earley(ParserBase):

    """Earley parser is able to parse any Context-Free Language properly.

    However the parsing process is slow due to the on-the-fly calculation
    of item set closures, where the detection of items which also includes
    the stack for arguments is very expensive.

    The construction of Earley parse trees is not quite straightforward.

    Generally schemes:
    - Chart structure
    - Item pointers
    """

    def __init__(self, grammar):
        super(Earley, self).__init__(grammar)
        self.reset()

    def reset(self):
        pass

    def parse_states(self, inp: str):
        """On-the-fly parsing maintaining states.
        """
        G = self.grammar
        S = self.states = []
        T = self.tokens = []

        # Forest F[k] is a dict, of which
        # - a key is a nonterminal
        # - a value is the set of rules with associated subtree stacks
        #
        # Suppose there are two rules in grammar for A, (A -> .αBγ) and (A -> μ),
        # then
        # {(A -> .αBγ): [[]], 
        #  (A -> .μ)  : [[]]}
        # is in F[k]
        F = self.forest = []
        F0 = defaultdict(list)
        top = G.start_rule
        F0[top].append((ParseTree(top, []), list(top.rhs)))
        F.append(F0)

        S.append([(0, G.make_item(0, 0))])

        for k, tok in enumerate(G.tokenize(inp, with_end=False)):
            T.append(tok)
            F.append(defaultdict(list)) # forest[i] :: {<Rule>: list<pair<<args>, <rest>>>}

            at, lex, lexval = tok
            # Closuring current item set S[k]
            # ItemSet S[k] is the AGENDA in context of chart parsing
            C = S[-1]
            C1 = []

            # compers = defaultdict(list)
            z_j = 0
            while z_j < len(C):
                j, jtm = C[z_j]
                # ISSUES by grammars having ε-Rules:
                # 
                # - In grammar without nullable rules, each predicted
                #   non-kernel item consumes no further nonterminal
                #   symbols, which means each completion item has no
                #   effect on generated prediction items and can be
                #   processed only ONCE and merely SINGLE-PASS-CLOSURE is
                #   needed;
                # - BUT in grammar with nullable rules, each predicted
                #   non-kernel item can consume arbitrarily many
                #   further nonterminals which leads to its completion,
                #   meaning prediction may result in new completion items,
                #   leading to repeated processing of the identical
                #   completion item. Thus LOOP-closure is needed.
                # 
                if not jtm.ended():
                    # Prediction: find nonkernels
                    if jtm.actor in G.nonterminals:
                        for r, rule in enumerate(G.rules):
                            if rule.lhs == jtm.actor:
                                ktm = G.make_item(r, pos=0)
                                new = (k, ktm)
                                if new not in C:
                                    C.append(new)
                    # Scanning/proceed for next States
                    elif jtm.actor == lex:
                        C1.append((j, jtm.shifted))
                else: # jtm.ended() at k
                    for (i, itm) in S[j]:
                        if not itm.ended() and itm.actor == jtm.target:
                            new = (i, itm.shifted)
                            if new not in C:
                                C.append(new)
                z_j += 1
            if not C1:
                # raise ValueError('Choked by {} at {}.'.format(lexval, at))
                if lex != END:
                    print('Unrecognized token {} Ignored. '.format(tok))
            else:
                # Commit proceeded items (by Scanning) as item set.
                S.append(C1)

        # Final completion
        k = len(S) - 1
        C = S[k]
        z_j = 0
        while z_j < len(C):
            j, jtm = C[z_j]
            if jtm.ended():
                for z_i, (i, itm) in enumerate(S[j]):
                    if not itm.ended() and itm.actor == jtm.target:
                        new = (i, itm.shifted)
                        if new not in C:
                            C.append(new)
            z_j += 1

    def parse_chart(self, inp: str):
        """Perform chart-parsing combined with Earley's recognition
        algorithm. The result is the chart, which semantically
        includes all possible parse trees w.R.t. the input.

        Parameters:
            :inp: The input string to be parsed.
        Returns:
            None

        Semantics of local variables:

        :tokens: Caching the input tokens;

        :chart: The graph recording all recognitions, where
        chart[j][k] is a set containing all recognized parsings on
        input segment from j'th to k'th tokens.

        :agenda: During the Earley parsing process, the agenda caches
        all items that should be processed when consuming the k'th
        input token.

        """
        G = self.grammar
        # chart[i][j] registers proceeding items from @i to @j
        chart = self.chart = defaultdict(lambda: defaultdict(set))
        # forest[i][j] registers completed parse trees from @i to @j
        # - Trees are constructed on-the-fly
        #   - Scanning Token T
        #     - index a ParseLeaf in forest[k][k+1]
        #     - For each item acting on T, mutate it with a new subtree ParseLeaf
        #   - if the current (j, B -> y, k) being processed is completed
        #     - index a ParseTree in forest[j][k]
        #     - 
        #   
        forest = self.forest = defaultdict(
            lambda: defaultdict(
                lambda: default(set)))
        tokens = self.tokens = []
        agenda = [(0, G.make_item(0, 0))]
        for k, tok in enumerate(G.tokenize(inp, False)):
            tokens.append(tok)
            at, lex, lexval = tok
            agenda1 = []
            while agenda:
                # A item `jtm` from position @j
                j, jtm = agenda.pop()
                # Directly register intermediate state item
                if jtm not in chart[j][k]:
                    chart[j][k].add(jtm)
                    # Forest: conclude subtree @j
                if not jtm.ended():
                    # Prediction
                    if jtm.actor in G.nonterminals:
                        for r, rule in enumerate(G.rules):
                            if rule.lhs == jtm.actor:
                                ktm = G.make_item(r, 0)
                                if ktm not in chart[k][k]:
                                    chart[k][k].add(ktm)
                                    agenda.append((k, ktm))
                    # Scanning
                    elif jtm.actor == lex:
                        # chart[k][k+1].add(jtm.shifted)
                        chart[j][k+1].add(jtm.shifted)
                        agenda1.append((j, jtm.shifted))
                        # Forest: conclude Leaf @k
                        # Log subtree for all items @j waiting this Token
                else:
                    # Completion
                    for i in range(j+1):
                        if j in chart[i]:
                            for itm in chart[i][j]:
                                if not itm.ended() and itm.actor == jtm.target:
                                    if itm.shifted not in chart[i][k]:
                                        agenda.append((i, itm.shifted))
                                        # Each tree initialized at @i gets a subtree for `jtm`
                                        # Log subtree for all items @i waiting this
                                        # nonterminal
            if agenda1:
                agenda = agenda1
            else:
                raise ValueError('Fail: empty new agenda by {}\nchart:\n{}'.format(tok,
                    pp.pformat(chart)))

        # Final completion
        k = len(tokens)
        while agenda:
            j, jtm = agenda.pop()
            if jtm.ended():
                chart[j][k].add(jtm)
                for i in range(j+1):
                    if j in chart[i]:
                        for itm in chart[i][j]:
                            if not itm.ended() and itm.actor == jtm.target:
                                if itm.shifted not in chart[i][k]:
                                    agenda.append((i, itm.shifted))
        # Algo finished.

    def parse_forest(self, inp: str):
        """Construct single-threaded parse trees during computing Earley
        states.

        """
        G = self.grammar
        # state :: {<Item>: [<Stack>]}
        s0 = {(0, G.make_item(0, 0)): [()]}
        ss = self.forest = []
        ss.append(s0)

        for k, tok in enumerate(G.tokenize(inp, with_end=True)):

            s_acc = ss[-1]
            s_act = {**s_acc}
            s_after = {}

            while 1:
                s_aug = {}
                for (j, jtm), j_stks in s_act.items():
                    # prediction
                    # TODO: Support nullable grammar (without LOOP) by applying
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
                                    if new in s_aug:
                                        tar = s_aug[new]
                                    else:
                                        tar = s_aug[new] = []
                                    for i_stk in i_stks:
                                        tar.append(i_stk + (j_tr,))
                if s_aug:
                    # Update, but NO replacement!!
                    # !Wrong: s_acc.update(s_aug)
                    for (i, itm), i_stk in s_aug.items():
                        if (i, itm) in s_acc:
                            s_acc[i, itm].extend(i_stk)
                        else:
                            s_acc[i, itm] = i_stk
                    s_act = s_aug
                else:
                    break

            if tok.symbol != END:
                ss.append(s_after)

        # 
        res = []
        for (i, itm), i_stks in ss[-1].items():
            if i == 0 and itm.r == 0 and itm.ended():
                for i_stk in i_stks:
                    assert len(i_stk) == 1, 'Top rule should be Singleton rule.'
                    res.append(i_stk[0]) 

        return res

    # HOWTO: Construct a parse forest with given chart. 
    # def find_subtrees(self, rule, i0, k):
    #     chart = self.chart
    #     tokens = self.tokens
    #     G = self.grammar
    #     # a list of stacks
    #     stks = [(i0,)]
    #     # 
    #     for y in rule.rhs:
    #         nstks = []
    #         # nstks = set()
    #         for stk in stks:
    #             i = stk[-1]
    #             for j in chart[i]:
    #                 for itm in chart[i][j]:
    #                     if y in G.nonterminals and itm.ended() and itm.target == y:
    #                         for node in self.find_trees(y, i, j):
    #                             nstks.append(stk[:-1] + (node, j,))
    #                             # nstks.add(stk[:-1] + (node, j,))
    #                     if y in G.terminals and i+1 == j and itm.prev == y:
    #                         nstks.append(stk[:-1] + (tokens[i], j,)) 
    #                         # nstks.add(stk[:-1] + (tokens[i], j,)) 
    #         stks = nstks
    #     for stk in stks:
    #         if stk[-1] == k:
    #             yield stk[:-1]

    # def find_trees(self, X, i, k=None):
    #     chart = self.chart
    #     tokens = self.tokens
    #     G = self.grammar
    #     #
    #     if k is None:
    #         k = len(tokens)
    #     # 
    #     nodes = []
    #     assert X in G.nonterminals
    #     for itm in chart[i][k]:
    #         if itm.ended() and itm.target == X:
    #             for subs in self.find_subtrees(itm.rule, i, k):
    #                 nodes.append(ParseTree(itm.rule, subs))
    #     return nodes

    def parse(self, inp: str):
        """Fully parse. Reset states. Deliver fully parsed result as a list of parse trees. """
        # Problem: how to traverse the chart to retrieve underlying parse trees?
        #
        # if there is a parse (A -> BC) in chart[i][k]
        # then there is some j, that (B -> x)
        #
        #
        
        self.reset()
        self.parse_chart(inp)
        # How to build parse forest from chart?
        chart = self.chart
        # DFS building with stack forking
        #
        tokens = self.tokens
        G = self.grammar
        

@meta
class LALR(ParserBase):

    """
    Extend base class Grammar into a LALR parser. The CLOSURE algorithm differs
    from the default one defined in super class Grammar.
    """
    DUMMY = '\0'

    def __init__(self, grammar):
        super(LALR, self).__init__(grammar)
        self.calc_lalr_item_sets()

    def reset(self):
        self.sstack = [0]

    def calc_lalr_item_sets(self):

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
        for ctm, a in G.closure_with_lookahead(init_item, END):
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
                C = G.closure_with_lookahead(ktm, LALR.DUMMY)
                for ctm, a in C:
                    if not ctm.ended():
                        X = ctm.actor
                        j = goto[i][X]
                        if a != LALR.DUMMY:
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
                    for ctm, lk1 in G.closure_with_lookahead(itm, lk):
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
                msg = '\n'.join([
                    '! LALR-Conflict raised:',
                    '  - in ACTION[{}]: '.format(i),
                    '{}'.format(pp.pformat(ACTION[i])),
                    "  * conflicting action on token {}: ".format(repr(lk)),
                    "{{{}: ('reduce', {})}}".format(repr(lk), itm)
                ])
            msg = '\n########## Error ##########\n {} \n#########################\n'.format(msg)
            raise ValueError(msg)

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
        # while p_tok < len(toks):
        try:
            while 1:
                i = sstack[-1]
                # tok = toks[p_tok]
                at, lex, lexval = tok
                if lex not in ACTION[i]:
                    msg =  'LALR - Ignoring syntax error by {}'.format(tok)
                    msg += '\n  Current predi stack: {}'.format(sstack)
                    msg += '\n'
                    print(msg)
                    # p_tok += 1
                    tok = next(toker)

                else:
                    act, arg = ACTION[i][lex]

                    # SHIFT
                    if act == 'shift':
                        if interp:
                            trees.append(lexval)
                        else:
                            trees.append(tok)
                        sstack.append(GOTO[i][lex])
                        # Go on iteration/scanning
                        # p_tok += 1
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
                        raise ValueError('Invalid action {} on {}'.format(act, arg))

        except StopIteration:
            raise ValueError('No enough token for completing the parse. ')

    def interpret(self, inp):
        return self.parse(inp, interp=True)


@meta
class FLL1(ParserBase):
    """Fake-LL(1)-Parser.

    Since strong-LL(1) grammar parser includes the usage of FOLLOW
    set, which is only heuristically helpful for the recognitive
    capability when handling NULLABLE rules, this parser suppress
    the need of FOLLOW.

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
        super(FLL1, self).__init__(grammar)
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
        """Table-driven FLL1 parsing.

        Tracing parallel stacks, where signal (α -> A) means reducing
        production A with sufficient amount of arguments on the
        argument stack:

        tokens:          | aβd#
        predi stack:     | S#
        argstack:        | #

        ==(S -> aBc)==>>

        tokens:          | aβd#
        predi stack:     | aBd(aBc -> S)#
        argstack:        | #

        ==a==>>

        tokens:        a | βd#
        predi stack:   a | Bd(aBc -> S)#
        argstack:        | #a

        ==(prediction B -> β), push predicted elements into states ==>>

        tokens:        a | βd#
        predi stack:  aB | β(β -> B)d(aBc -> S)#
        argstack:        | #a β1 β2 ... βn

        ==(reduce B -> β), push result into args ==>>

        tokens:        a | d#
        predi stack:  aB | d(->S)#
        argstack:        | #<a> <B>

        and further.

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
#         return FLL1(grammar)


class SExp(metaclass=FLL1.meta):
    l1 = r'\('
    r1 = r'\)'
    symbol = r'[^\(\)\s]+'
    def SExp(symbol):
        return symbol
    def SExp(l1, SExps, r1):
        return SExps
    def SExps():
        return []
    def SExps(SExp, SExps):
        return [SExp] + SExps


# # pp.pprint(SExp.table)
# res = SExp.parse('( ( (a ab  xyz  cdef) just ))', interp=0)
# pp.pprint(res)
# # pp.pprint(res.to_tuple())
# pp.pprint(res.translate())
# # pp.pprint(SExp.parse('( ( (a ab  xyz  cdef) just ))', interp=1))
