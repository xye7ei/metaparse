"""
This module prepares an alternative utilities for syntactical analyzing
in pure Python environment. It includes:

- OO-style grammar object system for context-free languages with completeness check 
- Elegant grammar definer based upon metaprogramming
- Parser interface
- Optional parsing algorithms
  - LL(1)
  - Earley
  - LALR

"""

import re
import inspect
import pprint as pp

from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple


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
                n = False
                for _, rhs in G[X]:
                    if rhs:
                        path += (X,)
                        if all(nullable(Y, path) for Y in rhs):
                            NULLABLE.add(X)
        for n in ns:
            nullable(n)

        FIRST = {}           # :: Dict[<nonterminal>: Set[<terminal>]]
        def first(X, path=()):
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
                        F.update(first(Y, path))
                        if Y not in NULLABLE:
                            break
                if X not in FIRST:
                    FIRST[X] = F
                else:
                    FIRST[X].update(F)
                return F

        for n in ns:
            first(n)

        G.NULLABLE = NULLABLE
        G.FIRST = FIRST
        G.first = first

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
                        beta = itm.over_rest()
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
                            for b in G.first(X):
                                if b is not EPSILON and b not in jlk:
                                    jlk.append(b)
                            if EPSILON not in G.first(X):
                                break
                        for b in jlk:
                            jtm = G.make_item(j, 0)
                            if (jtm, b) not in C:
                                C.append((jtm, b))
            z += 1
        return C


class ParserBase(object):

    """Abstract class for parsers. """

    def __init__(self, grammar):
        self.grammar = grammar

    def __repr__(self):
        # raise NotImplementedError('Parser should override __repr__ method. ')
        return self.__class__.__name__ + 'Parser-{}'.format(self.grammar)

    def parse(self):
        raise NotImplementedError('Any parse should have a `parse` method.')


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
        return [v for k, v in self if k == k0]


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
            # Handle normal attributes.
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
        self.evaluated_subs = []

    def __repr__(self):
        return 'ParseTree`{}`{}'.format(self.rule, pp.pformat(self.subs))

    def completed(self):
        return not self.subs

    def eval(self):
        args = []
        for sub in self.subs:
            if isinstance(sub, ParseTree):
                args.append(sub.eval())
            else:
                args.append(sub)
        return self.rule.seman(*args)

    def eval_traverse(tree):
        stack = []
        # state in stack :: (<function>, <evaled-args>, <unevaled-args>)
        stack.append(tree)
        while stack:
            tr = stack.pop()
            if isinstance(tree, ParseTree):
                # Conclude popped subtree as value
                if not tr.subs:
                    val = tr.rule.seman(*tr.evaluated_subs)
                    if not stack:
                        return val
                    else:
                        stack[-1].evaluated_subs.append(val)
                # Push back and transform subtree
                else:
                    stack.append(tr)
                    sub = tr.subs.pop(0) # Option: parseTree instance should not be modified!?!?!
                    if isinstance(sub, ParseLeaf):
                        tr.evaluated_subs.append(sub.value)
                    else: # isinstance(sub, ParseTree)
                        stack.append(sub)
            else:
                raise ValueError('stack must contain only ParseTrees')
            # elif isinstance(tree, ParseLeaf):
            #     assert stack, 'stack not empty'
        raise ValueError('Eval error')



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
        self.tokens = []
        self.graph = defaultdict(lambda: defaultdict(set))
        self.states = [[(self.grammar.make_item(r=0, pos=0), 0)]]

    def parse_feed(self, inp: str):
        """On-the-fly parsing maintaining states.
        """
        G = self.grammar
        S = self.states
        T = self.tokens
        graph = self.graph

        for k, tok in enumerate(G.tokenize(inp, with_end=False)):
            T.append(tok)
            at, lex, lexval = tok
            # Closuring current item set S[k]
            # ItemSet S[k] is the AGENDA in context of chart parsing
            C = S[-1]
            C1 = []

            z_j = 0
            while z_j < len(C):
                jtm, j = C[z_j]
                graph[j][k].add(jtm)
                if not jtm.ended():
                    # Prediction: find nonkernels
                    if jtm.actor in G.nonterminals:
                        for r, rule in enumerate(G.rules):
                            if rule.lhs == jtm.actor:
                                ktm = G.make_item(r, pos=0)
                                # edge for PUSHDOWN
                                graph[k][k].add(ktm)
                                new = (ktm, k)
                                if new not in C:
                                    C.append(new)
                    # Scanning/proceed for next States
                    elif jtm.actor == lex:
                        # Token with whole token information or only literal token-value?
                        # Also proceeded item should be added in!!
                        k1tm = jtm.shifted
                        graph[j][k+1].add(k1tm)
                        graph[k][k+1].add(k1tm)
                        C1.append((k1tm, j))
                # Completion
                else: # jtm.ended() at k
                    for z_i, (itm, i) in enumerate(S[j]):
                        if not itm.ended() and itm.actor == jtm.target:
                            graph[i][k].add(itm.shifted)
                            new = (itm.shifted, i)
                            if new not in C:
                                C.append(new)
                z_j += 1
            if not C1:
                # raise ValueError('Choked by {} at {}.'.format(lexval, at))
                print('Unrecognized token {} Ignored. '.format(tok))
            else:
                # Commit proceeded items (by Scanning) as item set.
                S.append(C1)

        # Final completion
        k = len(S) - 1
        C = S[k]
        z_j = 0
        while z_j < len(C):
            jtm, j = C[z_j]
            graph[j][k].add(jtm)
            if jtm.ended():
                for z_i, (itm, i) in enumerate(S[j]):
                    if not itm.ended() and itm.actor == jtm.target:
                        graph[i][k].add(itm.shifted)
                        new = (itm.shifted, i)
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
        chart = self.chart = defaultdict(lambda: defaultdict(set))
        agenda = [(0, G.make_item(0, 0))]
        tokens = []
        for k, tok in enumerate(G.tokenize(inp, False)):
            tokens.append(tok)
            at, lex, lexval = tok
            agenda1 = []
            while agenda:
                j, jtm = agenda.pop()
                chart[j][k].add(jtm)
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
                        chart[k][k+1].add(jtm.shifted)
                        chart[j][k+1].add(jtm.shifted)
                        agenda1.append((j, jtm.shifted))
                else:
                    # Completion
                    # if jtm not in chart[j][k]:
                    for i in range(j+1):
                        if j in chart[i]:
                            for itm in chart[i][j]:
                                if not itm.ended() and itm.actor == jtm.target:
                                    if itm.shifted not in chart[i][k]:
                                        # chart[i][k].add(itm.shifted)
                                        # agenda.append((j, jtm)) # for reuse: `jtm can complete more than one `itm
                                        agenda.append((i, itm.shifted))
            if agenda1:
                agenda = agenda1
            else:
                raise ValueError('Fail: empty new agenda by {}\nchart:\n{}'.format(tok,
                    pp.pformat(chart)))
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
                                    # chart[i][k].add(itm.shifted)
                                    agenda.append((i, itm.shifted))

    def parse(self, inp: str):
        """Fully parse. Reset states. Deliver fully parsed result as a list of parse trees. """
        self.reset()
        self.parse_feed(inp)


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
                    '{}'.format(pp.pformat(ACTION[i], indent=4)),
                    "  * conflicting action on token {}: ".format(repr(lk)),
                    "    {{{}: ('reduce', {})}}".format(repr(lk), itm)
                ])
            msg = '\n########## Error ##########\n {} \n#########################\n'.format(msg)
            raise ValueError(msg)

        self.ACTION = ACTION

    def parse(self, inp: str, interp=False):

        """Perform table-driven deterministic parsing process. Only one parse
        tree is to be constructed.

        Full-Parse with this process. On-the-fly parsing has a lag of reacting
        to active symbol.

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
                    msg += '\n  Current state stack: {}'.format(sstack)
                    msg += '\n'
                    print(msg)
                    # p_tok += 1
                    tok = next(toker)

                else:
                    act, arg = ACTION[i][lex]

                    # SHIFT
                    if act == 'shift':
                        trees.append(lexval)
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
                            tree = (ntar, subts)
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



class earley(cfg): 
    def __new__(mcls, n, bs, kw):
        grammar = cfg.__new__(mcls, n, bs, kw)
        return Earley(grammar)


class lalr(cfg): 
    def __new__(mcls, n, bs, kw):
        grammar = cfg.__new__(mcls, n, bs, kw)
        return LALR(grammar)


