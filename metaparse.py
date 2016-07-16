# -*- coding: utf-8 -*-

import re
import warnings
import inspect
import ast
import pprint as pp

from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple
from collections import deque
from collections import Iterable


class GrammarWarning(UserWarning):
    """Any way to specify user warning? """
    pass


class GrammarError(Exception):
    """Specifies exceptions raised when constructing a grammar."""
    pass


class ParserWarning(UserWarning):
    """Any way to specify user warning? """
    pass


class ParserError(Exception):
    """Specifies exceptions raised when error occured during parsing."""
    pass


# Special lexical element and lexemes
END = 'END'
END_PATTERN_DEFAULT = r'$'

IGNORED = 'IGNORED'
IGNORED_PATTERN_DEFAULT = r'[ \t\n]'

ERROR = 'ERROR'
ERROR_PATTERN_DEFAULT = r'.'

PRECEDENCE_DEFAULT = 1

class Symbol(str):

    """Symbol is a subclass of :str: with unquoted representation."""

    def __repr__(self):
        return self


class Signal(object):

    """Signal is used to direct parsing actions.

    """
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


DUMMY   = Symbol('\0')
EPSILON = Symbol('ε')

PREDICT = Signal('PREDICT')
SHIFT   = Signal('SHIFT')
REDUCE  = Signal('REDUCE')
ACCEPT  = Signal('ACCEPT')


class Token(object):

    """Token object is the lexical element of a grammar, which also
    includes the lexeme's position and literal value.

    """

    def __init__(self, at, symbol, value):
        self.at = at
        self.symbol = symbol
        self.value = value

    def __repr__(self):
        return '({}: {})@[{}:{}]'.format(
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

    """

    def __init__(self, func):
        self.lhs = func.__name__
        self.rhs = []
        # Signature only works in Python 3
        # for x in inspect.signature(func).parameters:
        for x in inspect.getargspec(func).args:
            # Tail digital subscript like xxx_4
            s = re.search(r'_(\d+)$', x)
            # s = re.search(r'_?(\d+)$', x)
            if s:
                x = x[:s.start()]
            self.rhs.append(x)
        self.seman = func
        # Make use of annotations?
        # self.anno = func.__annotations__

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
    def raw(lhs, rhs, seman):
        rl = Rule(seman)
        rl.lhs = lhs
        rl.rhs = list(rhs)
        return rl

    def eval(self, *args):
        return self.seman(*args)

    @property
    def size(self):
        return len(self.rhs)

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

    def __init__(self, lexes, rules, attrs, prece=None):
        """A grammar object containing lexical rules, syntactic rules and
        associating semantics.

        Parameters:

        :lexes:  : odict{str<terminal-name> : str<terminal-pattern>}
            A ordered dict representing lexical rules.

        :rules:  : [Rule]
            A list of grammar rules.

        :attrs:
            A list of extra attributes/methods

        :prece:
            A dict of operator's precedence numbers

        Notes:

        * Checks whether there is a singleton TOP-rule and add such
          one if not.

        * Needs to check validity and completeness!!
            * Undeclared tokens;
            * Undeclared nonterminals;
            * Unused tokens;
            * Unreachable nonterminals/rules;
            * FIXME: Cyclic rules; ???

        """

        # Operator precedence table may be useful.
        self.OP_PRE = dict(prece) if prece else {}

        # odict{lex-name: lex-re}
        self.terminals = OrderedDict()
        for tmn, pat in lexes.items():
            # Name trailing with integer subscript indicating
            # the precedence.
            self.terminals[tmn] = re.compile(pat, re.MULTILINE)

        # [<unrepeated-nonterminals>]
        self.nonterminals = []
        for rl in rules:
            if rl.lhs not in self.nonterminals:
                self.nonterminals.append(rl.lhs)

        # Sort rules with consecutive grouping of LHS.
        rules.sort(key=lambda r: self.nonterminals.index(r.lhs))

        # This block checks completion of given grammar.
        unused_t = set(self.terminals).difference([IGNORED, END, ERROR])
        unused_nt = set(self.nonterminals).difference([rules[0].lhs]) 
        # Raise grammar error in case any symbol is undeclared
        msg = ''
        for r, rl in enumerate(rules):
            for j, X in enumerate(rl.rhs):
                unused_t.discard(X)
                unused_nt.discard(X)
                if X not in self.terminals and X not in self.nonterminals:
                    msg += '\nUndeclared symbol:'
                    msg += "@{}`th symbol '{}' in {}`th rule {}.".format(j, X, r, rl)
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

    def __getitem__(self, k):
        if k in self.ngraph:
            return self.ngraph[k]
        else:
            raise ValueError('No such LHS {} in grammar.'.format(k))

    def item(G, r, pos):
        """Create a pair of integers indexing the rule and active position.

        """
        return Item(G.rules, r, pos)

    def rules_start_with(G, sym):
        """Yield enumerated matching rules."""
        for r, rl in enumerate(G.rules):
            if rl.lhs == sym:
                yield r, rl

    def _calc_first_and_nullable(G):
        """Calculate the FIRST set of this grammar as well as the NULLABLE
        set. Transitive closure algorithm is applied.

        """
        ns = set(G.nonterminals)

        NULLABLE = {X for X, rhs in G.rules if not rhs}     # :: Set[<terminal>]
        def nullable(X, path=()):
            if X in ns and X not in path:
                for lhs, rhs in G[X]:
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

        # Calculate FIRST in iterative way?
        # 
        # - Consider the construction of a prediction-tree in the
        #   context of building a GLL parser

        for n in ns:
            first1(n)

        G.NULLABLE = NULLABLE
        G.FIRST = FIRST
        G.first1 = first1

    def first_star(G, seq, tail=DUMMY):
        """Find FIRST set of a sequence of symbols.

        :seq:   A list of strings

        """
        assert not isinstance(seq, str)
        fs = set()
        for X in seq:
            fs.update(G.first1(X))
            if EPSILON not in fs:
                return fs
        fs.discard(EPSILON)
        # Note :tail: can also be EPSILON
        fs.add(tail)
        return fs

    def tokenize(G, inp, with_end):
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
            for lex, rgx in G.terminals.items():
                m = rgx.match(inp, pos=pos)
                # The first match with non-zero length is yielded.
                if m and len(m.group()) > 0:
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
                for j, jrl in G.rules_start_with(itm.actor):
                    jtm = G.item(j, 0)
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
                # for j, jrl in enumerate(G.rules):
                #     if itm.actor == jrl.lhs:
                for j, jrl in G.rules_start_with(itm.actor):
                    for b in G.first_star(itm.look_over, a):
                        jtm = G.item(j, 0)
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

        lexes = []
        lexpats = []
        rules = []
        prece = {}
        attrs = []

        for lst in lsts:
            for k, v in lst:
                # Built-ins are of no use
                if k.startswith('__') and k.endswith('__'):
                    continue
                # Handle implicit rule declaration through methods.
                elif callable(v):
                    r = Rule(v)
                    if r in rules:
                        raise GrammarError('Repeated declaration of Rule {}.\n{}'.format(r, r.src_info))
                    rules.append(r)
                # Lexical declaration without precedence
                elif isinstance(v, str):
                    if k in lexes:
                        raise GrammarError('Repeated declaration of lexical symbol {}.'.format(k))
                    lexes.append(k)
                    lexpats.append(v)
                # With precedence
                elif isinstance(v, tuple):
                    assert len(v) == 2, 'Lexical pattern as tuple should be of length 2.'
                    assert isinstance(v[0], str) and isinstance(v[1], int), \
                        'Lexical pattern as tuple should be of type (str, int)'
                    if k in lexes:
                        raise GrammarError('Repeated declaration of lexical symbol {}.'.format(k))
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

        return Grammar(OrderedDict(zip(lexes, lexpats)), rules, attrs, prece)

    @classmethod
    def __prepare__(mcls, n, bs, **kw):
        return assoclist()

    def __new__(mcls, n, bs, accu):
        """After gathering definition of grammar elements, reorganization
        and first checkings are done in this method, resulting in creating
        a Grammar object.

        """

        return cfg.read_from_raw_lists(accu)

    @staticmethod
    def extract_list(decl):

        """Retrieve structural information of the function :decl: (i.e. the
        body and its components IN ORDER). Then transform these
        information into a list of lexical elements and rules as well
        as semantics.

        """

        # The context for the rule semantics is :decl:'s belonging
        # namespace, here :__globals__:.
        glb_ctx = decl.__globals__

        # The local context for registering a function definition
        # by :exec:.
        lcl_ctx = {}

        # Parse the source.
        t = ast.parse(inspect.getsource(decl))

        # Something about :ast.parse: and :compile: with
        # - literal string code;
        # - target form
        # - target mode
        # e = ast.parse("(3, x)", mode='eval')
        # e = compile("(3, x)", '<ast>', 'eval')
        # s = ast.parse("def foo(): return 123", mode='exec')
        # s = compile("def foo(): return 123", '<ast>', 'exec')

        lst = []
        # assi_objs = []

        for obj in t.body[0].body:

            if isinstance(obj, ast.Assign):
                # assi_objs.append(obj)
                k = obj.targets[0].id
                ov = obj.value
                # :ast.literal_eval: evaluates an node instantly!
                v = ast.literal_eval(ov)
                lst.append((k, v))
                # Some context values may be used for defining
                # following functions.
                lcl_ctx[k] = v

            elif isinstance(obj, ast.FunctionDef):
                name = obj.name
                # Conventional fix
                ast.fix_missing_locations(obj)
                # :ast.Module: is the unit of program codes.
                # FIXME: Is :md: necessary??
                md = ast.Module(body=[obj])
                # Compile a module-ast with '<ast>' mode targeting
                # :exec:.
                code = compile(md, '<ast>', 'exec')
                # Register function into local context.
                exec(code, glb_ctx, lcl_ctx)
                func = lcl_ctx.pop(name)
                lst.append((name, func))

            else:
                # Ignore structures other than :Assign: and :FuncionDef:
                pass

        return lst

    @staticmethod
    def v2(func):
        lst = cfg.extract_list(func)
        return cfg.read_from_raw_lists(lst)

grammar = cfg.v2

"""
The above parts are utitlies for grammar definition and extraction methods
of grammar information.

The following parts supplies utilities for parsing.
"""

# The object ParseLeaf is semantically indentical to the object Token.
ParseLeaf = Token


class ParseTree(object):
    """ParseTree is the basic object representing a valid parsing.

    Note the node value is a :Rule: object rather than a grammar
    symbol, which allows the tree to be traversed for translation with
    the rule's semantics.

    """

    def __init__(self, rule, subs):
        # Contracts
        assert isinstance(rule, Rule)
        for sub in subs:
            assert isinstance(sub, ParseTree) or isinstance(sub, ParseLeaf)
        self.rule = rule
        self.subs = subs

    def translate(tree, trans_tree=None, trans_leaf=None):
        """Postorder tree traversal with double-stack scheme.

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
        # Direct translation
        if not trans_tree:
            trans_tree = lambda t, args: t.rule.seman(*args)
        if not trans_leaf:
            trans_leaf = lambda tok: tok.value
        # Aliasing push/pop, simulating stack behavior
        push, pop = list.append, list.pop
        sstk = [(PREDICT, tree)]
        astk = []
        while sstk:
            act, t = pop(sstk)
            if act == REDUCE:
                args = []
                for _ in t.subs:
                    push(args, pop(astk))
                # apply semantics
                push(astk, trans_tree(t, args[::-1]))
            elif isinstance(t, ParseLeaf):
                push(astk, trans_leaf(t))
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
        """Translate the parse tree into Python tuple form. """
        return self.translate(lambda t, subs: (Symbol(t.rule.lhs), subs),
                              lambda tok: tok)

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

    """Abstract class for both deterministic/non-deterministic parsers.
    They both have methods :parse_many: and :interpret_many:, while
    non-deterministic parsers retrieve singleton parse list with these
    methods.

    """

    def __init__(self, grammar):
        assert isinstance(grammar, Grammar)
        self.grammar = grammar
        # Share auxiliary methods declared in Grammar instance
        for k, v in grammar.attrs:
            # assert k.startswith('_')
            setattr(self, k, v)
        # Delegation
        self.tokenize = grammar.tokenize

    def __repr__(self):
        # Polymorphic representation without overriding
        # raise NotImplementedError('Parser should override __repr__ method. ')
        return self.__class__.__name__ + '-Parser-{}'.format(self.grammar)

    def parse_many(self, inp, interp=False):
        # Must be overriden
        raise NotImplementedError('Any parser should have a parse method.')

    def interpret_many(self, inp):
        return self.parse_many(inp, interp=True)

    def parse1(self, inp, interp=False):
        res = self.parse_many(inp, interp)
        if res:
            return res[0]
        else:
            raise ParserError("No parse.")

    def interpret1(self, inp):
        return self.parse1(inp, interp=True)


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
        raise NotImplementedError('Deterministic parser should have parse1 method.')

    def interpret(self, inp):
        return self.parse(inp, interp=True)


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

    WARNING: In case with grammar having NULLABLE-LOOPs like:

      A -> A B
      A ->
      B -> b
      B -> 

    where A => ... => A consuming no input tokens, the on-the-fly
    computation of parse trees will not terminate. However,
    recognition without building parse trees stays intact.

    """

    def __init__(self, grammar):
        super(Earley, self).__init__(grammar)

    def recognize(self, inp):
        """Naive Earley's recognization algorithm. No parse produced.

        """
        G = self.grammar
        S = [[(0, G.item(0, 0))]]

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
                #   LOOP-closure is needed. But this can be avoided
                #   with an enhanced predictor which can predict one
                #   more symbol over a nullable actor.
                #
                if not jtm.ended():
                    # Prediction/Completion no more needed when
                    # recognizing END-Token
                    if not tok.is_END():
                        # Prediction: find nonkernels
                        if jtm.actor in G.nonterminals:
                            for r, rule in G.rules_start_with(jtm.actor):
                                ktm = G.item(r, pos=0)
                                new = (k, ktm)
                                if new not in C:
                                    C.append(new)
                                if not rule.rhs:
                                    # Directly NULLABLE
                                    enew = (j, jtm.shifted)
                                    if enew not in C:
                                        C.append(enew)
                        # Scanning/proceed for the next State
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

    def parse_chart(self, inp):
        """Perform general chart-parsing framework method with Earley's
        recognition algorithm. The result is a chart, which
        semantically includes Graph Structured Stack i.e. all possible
        parse trees.

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
        # chart[i][j] registers active items from @i to @j
        chart = self.chart = defaultdict(lambda: defaultdict(set))
        agenda = [(0, G.item(0, 0))]

        for k, tok in enumerate(G.tokenize(inp, True)):
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
                            for r, rule in G.rules_start_with(jtm.actor):
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
        s0 = {(0, G.item(0, 0)): [()]}
        ss = self.forest = []
        ss.append(s0)

        # Tokenizer with END token to force the tailing completion
        # pass.
        for k, tok in enumerate(G.tokenize(inp, with_end=True)): 
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
                            for r, rule in G.rules_start_with(jtm.actor):
                                new = (k, G.item(r, 0))
                                if new not in s_acc:
                                    if rule.rhs:
                                        s_aug[new] = [()]
                                    else:
                                        # Nullable completion and
                                        # prediction.
                                        new0 = (j, jtm.shifted)
                                        j_tr = ParseTree(rule, [])
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
                    iacts[REDUCE].append(itm)
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
                iacts[SHIFT][X] = j
                igoto[X] = j

            ACTION.append(iacts)
            GOTO.append(igoto)

            k += 1

    def parse_many(self, inp, interp=False):

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
            reds = ACTION[st][REDUCE]
            shif = ACTION[st][SHIFT]

            # REDUCE
            # There may be multiple reduction options. Each option leads
            # to ONE NEW FORK of the parsing thread.
            for ritm in reds:
                # Forking, copying State-Stack and trs
                # Index of input remains unchanged.
                frk = stk[:]
                trs1 = trs[:]

                subs = []
                for _ in range(ritm.size):
                    frk.pop()   # only serves amount
                    subs.insert(0, trs1.pop())

                # Deliver/Transit after reduction
                if ritm.target == G.top_rule.lhs:
                    # Discard partial top-tree, which can be legally
                    # completed while i < len(tokens)
                    if i == len(tokens):
                        # Not delivering augmented top tree
                        results.append(subs[0])
                else: 
                    trs1.append(ParseTree(ritm.rule, subs))
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

        if interp:
            return [tree.translate() for tree in results]
        else:
            return results


@meta
class LALR(ParserDeterm):

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

        Ks = [[G.item(0, 0)]]   # Kernels
        goto = []

        # Calculate LR(0) item sets and goto
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

            # Register local :igoto: into global goto.
            goto.append({})
            for X, J in igoto.items():
                # The Item-Sets should be treated as UNORDERED! So
                # sort J to identify the Lists with same items.
                J = sorted(J, key=lambda i: (i.r, i.pos))
                if J not in Ks:
                    Ks.append(J)
                j = Ks.index(J)
                goto[i][X] = j

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

        self._propa_passes = b

        self.Ks = Ks
        self.GOTO = goto
        self.propa = propa
        self.table = table

        # Now all goto and lookahead information is available.

        # Construct ACTION table
        ACTION = [{} for _ in table]

        # SHIFT for non-ended items
        # Since no SHIFT/SHIFT conflict exists, register
        # all SHIFT information firstly.
        for i, xto in enumerate(goto):
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
                        # with lookahead :lk1:
                        if ctm.ended():
                            new_redu = (REDUCE, ctm)
                            if lk1 in ACTION[i]:
                                # If there is already an action on
                                # :lk1:, test wether conflicts may
                                # raise for it.
                                act, arg = ACTION[i][lk1] 
                                # if the action is REDUCE
                                if act is REDUCE:
                                    # which reduces a different item :arg:
                                    if arg != ctm:
                                        # then REDUCE/REDUCE conflict is raised.
                                        conflicts.append((i, lk1, (act, arg), new_redu))
                                        continue
                                # If the action is SHIFT
                                else:
                                    # :ctm: is the item prepared for
                                    # shifting on :lk:
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
                                            # Trivially, maybe :op_lft: == :op_rgt:
                                            if G.OP_PRE[op_lft] >= G.OP_PRE[op_rgt]:
                                                # Left wins.
                                                ACTION[i][lk1] = new_redu
                                                continue
                                            else:
                                                # Right wins.
                                                continue
                                    # SHIFT/REDUCE conflict raised
                                    conflicts.append((i, lk1, (act, arg), (REDUCE, ctm)))
                            # If there is still no action on :lk1:
                            else: 
                                ACTION[i][lk1] = (REDUCE, ctm)
                # Accept-Item
                if itm.index_pair == (0, 1):
                    ACTION[i][END] = (ACCEPT, None)

        # Report LALR-conflicts, if any
        if conflicts:
            msg = "\n########## Error ##########"
            for i, lk, act0, act1 in conflicts:
                msg += '\n'.join([
                    '',
                    '! LALR-Conflict raised:',
                    '  - in state [{}]: '.format(i),
                    '{}'.format(pp.pformat(Ks[i])),
                    "  * conflict on lookahead {}: ".format(repr(lk)),
                    "{}".format({lk: [act0, act1]}),
                    # "{{{}: {}}}".format(repr(lk), act1),
                    '',
                ])
            msg += "############################"
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
        Ks = self.Ks
        sstack = self.sstack = [0]
        GOTO = self.GOTO
        ACTION = self.ACTION

        # Lazy extraction of tokens
        toker = self.grammar.tokenize(inp, with_end=True) # Use END to force finishing by ACCEPT
        tok = next(toker)
        warns = []

        try:
            while 1:

                # Peek state
                i = sstack[-1]

                if tok.symbol not in ACTION[i]:
                    msg = '\n'.join([
                        '',
                        'WARNING: ',
                        'LALR - Ignoring syntax error reading Token {}'.format(tok),
                        '- Current kernel derivation stack:\n{}'.format(
                            pp.pformat([Ks[i] for i in sstack])),
                        '- Expecting tokens and actions: \n{}'.format(
                            pp.pformat(ACTION[i])),
                        '- But got: \n{}'.format(tok),
                        '',
                    ])
                    warnings.warn(msg)
                    warns.append(msg)
                    if len(warns) == n_warns:
                        raise ParserError(
                            'Threshold of warning tolerance {} reached. Parsing exited.'.format(n_warns))
                    else:
                        tok = next(toker)

                else: 
                    act, arg = ACTION[i][tok.symbol]

                    # SHIFT
                    if act == SHIFT:
                        if interp:
                            trees.append(tok.value)
                        else:
                            trees.append(tok)
                        sstack.append(GOTO[i][tok.symbol])
                        # Go on scanning
                        tok = next(toker)

                    # REDUCE
                    elif act == REDUCE:
                        assert isinstance(arg, Item)
                        rtm = arg
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
                        # New symbol is used for shifting.
                        sstack.append(GOTO[sstack[-1]][rtm.target])

                    # ACCEPT
                    elif act == ACCEPT:
                        return trees[-1]

                    else:
                        raise ParserError('Invalid action {} on {}'.format(act, arg))

        except StopIteration:
            raise ParserError('No enough tokens for completing the parse. ')
            # pass

        return


@meta
class WLL1(ParserDeterm):
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
            raise ParserError('No enough token for complete parsing.')


@meta
class GLL(ParserBase):
    """Generalized LL(1) Parser.

    Honestly, GLL(1) Parser is just the prototype of Earley Parser
    with BFS nature but without integration of *Dynamic Programming*
    techniques (thus unable to detect left-recursion). The
    augmentation by memoizing recognition at certain postitions is
    quite straightforward.

    Since all LL(1) prediction conflicts in the table are traced by
    forking parallel stacks during parsing, left-sharing problem gets
    handled elegantly and correctly.

    """

    def __init__(self, grammar):
        super(GLL, self).__init__(grammar)
        self._calc_gll1_table()

    def _calc_gll1_table(self):
        G = self.grammar
        table = self.table = {}
        for r, rule in enumerate(G.rules):
            lhs, rhs = rule
            if lhs not in table:
                table[lhs] = defaultdict(list)
            if rhs:
                for a in G.first_star(rhs, EPSILON):
                    if a is not EPSILON:
                        table[lhs][a].append(rule)
            else:
                table[lhs][EPSILON].append(rule)

    def parse_many(self, inp, interp=False):
        """ 
        """ 
        # Tracing parallel stacks, where signal (α>A) means reducing
        # production A with sufficient amount of arguments on the
        # argument stack. 
        # | aβd#               :: {inp-stk}
        # | (#      ,      S#) :: ({arg-stk}, {pred-stk}) 
        # ==(S -> aBc), (S -> aD)==>> 
        # | aβd#
        # | (#      ,    aBd(aBd>S)#)
        #   (#      ,    aDe(aD>S)#) 
        # ==a==>> 
        # | βd#
        # | (#a     ,     Bd(aBd>S)#)
        #   (#a     ,     De(aD>S)#) 
        # ==prediction (B -> b h)
        #   prediction (D -> b m), push predicted elements into states ==>> 
        # | βd#
        # | (#a     ,bh(bh>B)d(aBd>S)#)
        #   (#a     ,bm(bm>D)De(aD>S)#) 
        # ==b==> 
        # | βd#
        # | (#ab    ,h(bh>B)d(aBd>S)#)
        #   (#ab    ,m(bm>D)De(aD>S)#) 
        # ==h==> 
        # | βd#
        # | (#abh   ,(bh>B)d(aBd>S)#)
        #   (#ab    ,{FAIL} m(bm>D)De(aD>S)#) 
        # ==(reduce B -> β), push result into args ==>> 
        # | βd#
        # | (#aB    ,d(aBd>S)#) 
        # and further. 
        # """

        push, pop = list.append, list.pop
        table = self.table
        G = self.grammar
        #
        agenda = [([], [(PREDICT, G.top_symbol)])]
        results = []
        #
        for k, tok in enumerate(G.tokenize(inp, with_end=True)):
            at, look, lexeme = tok
            # AGMENTATION AGENDA should be used!
            agenda1 = []
            while agenda:
                (astk, pstk) = agenda.pop(0)
                if not pstk and len(astk) == 1: # and tok.symbol == END:
                    # Deliver partial result?
                    if tok.is_END():
                        results.append(astk[0])
                else:
                    act, actor = pstk.pop(0)
                    # Prediction
                    if act is PREDICT:
                        if actor in G.nonterminals:
                            if look in table[actor]:
                                for rule in table[actor][look]:
                                    nps = [(PREDICT, x) for x in rule.rhs] + [(REDUCE, rule)]
                                    agenda.append((astk[:], nps + pstk))
                            # NULLABLE
                            if EPSILON in table[actor]:
                                erule = table[actor][EPSILON][0]
                                arg = ParseTree(erule, [])
                                agenda.append((astk + [arg], pstk[:]))
                        else:
                            assert isinstance(actor, str)
                            if look == actor:
                                astk.append(tok)
                                agenda1.append((astk, pstk))
                            else:
                                # # May report dead state here for inspection
                                # print('Expecting \'{}\', but got {}: \n{}\n'.format(
                                #     actor,
                                #     tok,
                                #     pp.pformat((astk, pstk))))
                                # # BUT this can be quite many!!
                                pass
                    # Completion
                    else:
                        assert isinstance(actor, Rule)
                        subs = []
                        for _ in actor.rhs:
                            subs.insert(0, astk.pop())
                        astk.append(ParseTree(actor, subs))
                        agenda.append((astk, pstk))
            if agenda1:
                agenda = agenda1

        if interp:
            return [res.translate() for res in results]
        else:
            return results


