import re
import pprint as pp

from collections import OrderedDict
from collections import defaultdict
from collections import namedtuple

END = 'END'
END_PAT = r'\Z'

IGNORED = 'IGNORED'
IGNORED_PAT = r'[ \t]'

ERR = 'ERR'
ERR_PAT = r'.'


Token = namedtuple('Token', 'at symb val')
Token.start = lambda s: s.at
Token.end = lambda s: s.at + len(s.val)
Token.__repr__ = lambda s: "Token('{}': {})@[{}:{}]".format(s.symb, s.val, s.at, s.end())


class Rule(object):

    """
    Rule object representing a rule in a Context Free Grammar.
    A Rule inside some specified scope should be constructed only once.
    The equality should not be overwritten, thus as identity to save
    performance when comparing rules while comparing some Item objects.
    """

    def __init__(self, func):
        self.lhs = func.__name__
        self.rhs = [x[:-1] if x[-1].isdigit() else x
                    for x in func.__code__.co_varnames]
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

STAR = '*'

@Rule
def E(E: STAR, PLUS, T: STAR) -> int:
    return E + T
    
class Item(object):

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

    def ended(s): return s.pos == len(s.rules[s.r].rhs)
    def rest(s): return s.rules[s.r].rhs[s.pos:]
    def over_rest(s): return s.rules[s.r].rhs[s.pos+1:]
    def active(s): return s.rules[s.r].rhs[s.pos]
    def shifted(s): return Item(s.rules, s.r, s.pos+1)
    def prev(s): return s.rules[s.r].rhs[s.pos-1]
    def size(s): return len(s.rules[s.r].rhs)
    def target(s): return s.rules[s.r].lhs


class Grammar(object):

    def __init__(self, lexes_rules):
        """
        lexes  :: {str<terminal/lex name> : str<lex re pattern>}
        rules  :: [Rule]

        Notes:
        - Automatically check whether there is a singleton TOP-rule
          and add such one if not.
        - Needs to perform validity and well-formity!!
            - Undeclared tokens;
            - Undeclared nonterminals;
            - Unused tokens;
            - Unreachable nonterminals/rules;
            - Cyclic rules;
        """
        lexes, rules = lexes_rules

        # {lex-name: lex-re}
        self.terminals = OrderedDict(
            (lex, re.compile(pat)) for lex, pat in lexes.items())

        # [non-repeated-nonterminals]
        self.nonterminals = []
        for rl in rules:
            if rl.lhs not in self.nonterminals:
                self.nonterminals.append(rl.lhs)

        # Perform validity check.
        _tm_unused = set(self.terminals)
        _tm_unused.discard(IGNORED)
        _tm_unused.discard(END)
        _tm_unused.discard(ERR)
        _nt_unused = set(self.nonterminals)
        _nt_unused.discard(rules[0].lhs)
        msg = ''
        for r, rl in enumerate(rules):
            for j, X in enumerate(rl.rhs):
                _tm_unused.discard(X)
                _nt_unused.discard(X)
                if X not in self.terminals and X not in self.nonterminals:
                    msg += '\nUndeclared symbol:'
                    msg += "@{}`th symbol '{}' in {}`th rule {}.".format(j, X, r, rl)
        if msg:
            msg += '\n...Failed to construct the parser.'
            raise ValueError(msg)
        for tm in _tm_unused:
            print('Warning: Unused terminal symbol {}'.format(tm))
        for nt in _nt_unused:
            print('Warning: Unused nonterminal symbol {}'.format(nt))

        # Generate top rule as Augmented Grammar only if
        # the top rule is not explicitly given.
        fst_rl = rules[0]
        if len(fst_rl.rhs) != 1 or 1 < [rl.lhs for rl in rules].count(fst_rl.lhs):
            # Consider symbol END...
            # For LALR method, the top rule
            # [S^ -> S END ; ??] should rather be
            # [S^ -> S     ; END]
            # That is, the first item set should be [[S^ -> .S ; END]]
            # thus to be consistent with calculating spontaneous/progagation
            # of LALR-Kernel, which is [[S^ -> .S ; '#']]
            tp_symb = "{}^".format(fst_rl.lhs)
            tp_rl = Rule.raw(
                tp_symb,
                [fst_rl.lhs], lambda x: x)
            self.nonterminals.insert(0, tp_symb)
            rules = [tp_rl] + rules

        self.rules = rules
        self.symbols = self.nonterminals + [a for a in lexes if a != END]

        # Top rule of Augmented Grammar.
        self.start_rule = self.rules[0]
        self.start_symbol = self.start_rule.lhs

        # Helper for fast accessing with Trie like structure.
        self.ntrie = defaultdict(list)
        for rl in self.rules:
            self.ntrie[rl.lhs].append(rl)

        # Prepare useful information for parsing.
        self._calc_first_and_nullable()

    def __repr__(self):
        return 'Grammar\n{}\n'.format(pp.pformat(self.rules))

    def __getitem__(self, k):
        return self.ntrie[k]

    def Item(G, r, pos):
        return Item(G.rules, r, pos)

    def _calc_first_and_nullable(G):
        """The computation of FIRST, NULLABLE and CLOSURE can be combined all
        in one process like follows, where recursion and memoization
        are utilized to avoid repeated computation.

        """
        ns = set(G.nonterminals)

        DESCEND = defaultdict(list) # :: Dict[<nonterminal>: Set[<nonterminal>]]
        FIRST = {}           # :: Dict[<nonterminal>: Set[<terminal>]]
        NULLABLE = set()     # :: Set[<terminal>]

        def first(X, path=()):
            if X in path:
                return {}
            elif X not in ns:
                return {X}
            elif X in FIRST:
                return FIRST[X]
            else:
                F = set()
                for _, rhs in G[X]:
                    if not rhs:
                        F.add(None)
                        NULLABLE.add(X)
                    else:
                        # Use list rather than set to preserve visiting
                        # order for easy reading.
                        if rhs[0] in ns and rhs[0] not in DESCEND[X]:
                            DESCEND[X].append(rhs[0])
                        p = 0
                        while p < len(rhs):
                            FY = first(rhs[p], path+(X,))
                            F.update(FY)
                            if None not in FY:
                                break
                            else:
                                p += 1
                        F.discard(None)
                        if p == len(rhs):
                            NULLABLE.add(X)
                            F.add(None)
                FIRST[X] = F
                return F

        for n in ns:
            first(n)

        G.NULLABLE = NULLABLE
        G.FIRST = FIRST
        G.first = first

    def tokenize(G, inp: str):
        """Perform lexical with given string. Yield feasible matches with
        defined lexical patterns. Ambiguity is resolved by the order
        of patterns within definition.

        """
        # How to modify a group of regexes into more efficient
        # Trie with each node as atomic regex thus to optimize preformance?

        pos = 0
        while pos < len(inp):
            for lex, rgx in G.terminals.items():
                m = rgx.match(inp, pos=pos)
                if m:
                    break
            if m and lex == IGNORED:
                _, pos = m.span()
            elif m and lex != IGNORED:
                at, pos = m.span()
                yield Token(at, lex, m.group())
            else:
                # Report error here!
                # print('Unrecognized token at {} with value {}!'.format(at, m.group()))
                yield Token(at, ERR, m.group())
                pos += 1
        yield Token(pos, END, END_PAT)

    def closure(G, I):
        """Naive CLOSURE calculation without lookaheads. """

        C = I[:]
        z = 0
        while z < len(C):
            itm = C[z]
            if not itm.ended():
                for j, jrl in enumerate(G.rules):
                    if itm.active() == jrl.lhs:
                        beta = itm.over_rest()
                        jtm = G.Item(j, 0)
                        if jtm not in C:
                            C.append(jtm)
            z += 1
        return C


class assoclist(list):

    """
    Assoclist :: [(k, v)]
    """

    def __setitem__(self, k, v):
        ls = super(assoclist, self)
        ls.append((k, v))

    def __getitem__(self, k0):
        return [v for k, v in self if k == k0]

# There are two ways to specify a parser for given Grammar:
# 
# - class G1(metaclass=cfg):
#   ......
#   lalrpar = LALR(*G1)
#   ......
#   - This serves for comparison of different Parsers,
#     since the grammar are shared by them and written
#     only once. 
# 
# - class lalrpar(metaclass=lalr):
#   ......
#   - This serves for single use of one specified Parser. 


class cfg(type):

    """This metaclass performs accumulation for all declared grammar
    symbols and rules in a instance class scope, where declared
    variables are registered as Lexical elements into a k-v dict and
    functions are registered as Rule objects defined above.

    The constructed object of this typeclass of is a tuple like
    (<lexicals>, <rules>) :: Tuple[OrderedDict[str : str], List[Rule]]

    """

    @classmethod
    def __prepare__(mcls, n, bs, **kw):

        accu = assoclist()

        # An attribute which is to be accessed by metaclass.
        # accu['__name__'] = 'grammar'
        # accu['__return__'] = None

        return accu

    def __new__(mcls, n, bs, accu):
        """
        Check declaration/usage information.
        Complete ordinal helpers.
        """

        lexes = OrderedDict()
        rules = []

        for k, v in accu:

            # Handle lexical declaration.
            if not k.startswith('_') and isinstance(v, str):
                lexes[k] = v

            elif isinstance(v, Rule):
                rules.append(v)

            # Handle rule declaration.
            elif not k.startswith('_') and callable(v):
                rules.append(Rule(v))

            # Handle normal attributes.
            else:
                pass

        # May be overwritten.
        # Consider default matching order for thess Lexers! 
        # Always match END first;
        # Always match IGNORED secondly;
        # Always match ERR at last;

        if IGNORED not in lexes:
            lexes[IGNORED] = IGNORED_PAT
            lexes.move_to_end(IGNORED, last=False)

        lexes[END] = END_PAT
        lexes.move_to_end(END, last=False)

        if ERR not in lexes:
            lexes[ERR] = ERR_PAT

        return (lexes, rules)
