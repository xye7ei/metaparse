import re
import warnings
import pprint as pp

from collections import namedtuple as data
from collections import OrderedDict as odict


Rule = data('Rule', 'lhs rhs seman')
Rule.__repr__ = lambda s: '( {} -> {} )'.format(s.lhs, ' '.join(map(str, s.rhs)))


Item = data('Item', 'rule i stk')
Item.__repr__ = lambda s: '[ {:<10} ->   {:<30} ]'.format(str(s.rule.lhs),
                                                     ' '.join(s.rule.rhs[:s.i]) \
                                                     + '.' + \
                                                     ' '.join(s.rule.rhs[s.i:]))
Item.make_shifted_with = lambda s, x: Item(s.rule, s.i + 1, s.stk + (x,))
Item.waiting = lambda s: None if s.i == len(s.rule.rhs) else s.rule.rhs[s.i]
Item.target = lambda s: s.rule.lhs
Item.value = lambda s: s.rule.seman(*s.stk)

def eq(a, b):
    return id(a) == id(b) or a == b

class Grammar(object): 


    """
    Abstraction of Context-Free Grammar.
    Open questions:
      - Need to register Nullable rules?
        - Automatically satisfied by the nature of Earley-completion. Since any atomic nullable
          Rule's corresponded Item is always completed. 
    """


    def __init__(self, A, N, R, n0):

        """
        A  :: {str<terminal/lex name> : str<lex re pattern>}
        N  :: {str<nonterminal name>}
        R  :: [Rule]
        n0 :: str<entrance symbol>
        """

        self.terminals = odict((lex, re.compile(pat)) for lex, pat in A.items())
        self.nonterminals = N
        self.rules = R 

        self.dict = {}
        for p in R:
            if p[0] not in self.dict:
                self.dict[p[0]] = []
            self.dict[p[0]].append(p)

        self.start_rule = self.dict[n0][0]
        self.start_item = Item(self.start_rule, i=0, stk=())


    def __repr__(self):
        return 'Grammar\n{}\n'.format(pp.pformat(self.rules))


    def lexing(self, inp):
        # How to modify a group of regexes into more efficient Trie with each node as
        # atomic regex thus to optimize preformance? 
        pos = 0
        while pos < len(inp):
            for lex, rgx in self.terminals.items():
                m = rgx.match(inp, pos=pos)
                if m: break
            if m and lex != 'IGNORED':
                at, pos = m.span()
                yield at, lex, m.group()
            elif lex == 'IGNORED':
                at, pos = m.span()
            else:
                # Discarding error??
                at += 1
                pos += 1
        yield at, 'END', '$'

    def earley(G, inp : str) -> [[Item]]: 

        """
        Parse a iterable of strings w.R.t. given grammar into states.
        - How to use the chart lazily??
        - How is the lazy overhead compared to strict?
          - Time?
          - Space?
        """ 

        S = [[(G.start_item, 0)]]

        for k, (at, lex, lexval) in enumerate(G.lexing(inp)):

            if not S[k]:
                raise ValueError('Failed at {}th position in input. '.format(at))
            t_waiters = []

            # ONE-PASS 
            # Note len(S[k]) is growing!
            z = 0
            while z < len(S[k]):

                jt, j = S[k][z]

                # prediction for `jt , i.e. applying CLOSURE
                if jt.waiting():
                    if jt.waiting() in G.nonterminals:
                        for prd in G.dict[jt.waiting()]:
                            new = (Item(prd, 0, ()), k)
                            if new not in S[k]:
                                S[k].append(new)
                    else:
                        t_waiters.append((jt, j))

                # completion with `jt
                else:
                    for it, i in S[j]:
                        if it.waiting():
                            if it.waiting() == jt.target():
                                # new = (it.make_shifted_with(jt), i)
                                new = (it.make_shifted_with(jt.value()), i)
                                if new not in S[k]:
                                    S[k].append(new)
                                    if new[0].waiting() in G.terminals:
                                        t_waiters.append(new)
                z += 1

            # ONE-PASS
            # scanning
            # S grows on the fly. Preparing new empty state list for scanning.
            S.append([]) 
            for jt, j in t_waiters:
                if jt.waiting() and jt.waiting() == lex:
                    S[k+1].append((jt.make_shifted_with(lexval), j)) 

        return S

    def parse(self, inp, method='earley'):
        # Lexer is waiting to be used to split input string...
        # Case matching...
        s = getattr(self, method)(inp)
        if s[-1]:
            return s, s[-1][0][0].value()
        else:
            return s, None

# Garith3.parser.parse('3 + 2 * 5')
# Garith3.parser.parse('3 + c * 5')


def rule(func):
    lhs = func.__name__
    rhs = [a[:-1] if a[-1].isdigit() else a
           for a in func.__code__.co_varnames]
    seman = func
    return Rule(lhs, rhs, seman)


class RuleAccu(dict):


    def __init__(self, *a, **kw):
        super(RuleAccu, self).__init__(self, *a, **kw)
        self.lxs   = odict()    # { str : re }
        self.ntms  = set()      # { str }
        self.rules = []         # [ rule ]
        
        # Waiting to be overwritten. 
        self.lxs['IGNORED'] = r' \t'
        self.lxs['END'] = r'\$'


    def __setitem__(self, k, v):
        """ Used as delegation while built-in system is collecting
        (shared) attributes within some scope. """

        # register Lex
        # All attributes without underscored start are treated as Lexicals. 
        if not k.startswith('_') and isinstance(v, str):
            # May override prepared lexer 'IGNORED' and 'END'
            self.lxs[k] = v

        # register Rule
        elif isinstance(v, Rule):
            self.ntms.add(k)
            self.rules.append(v)

        # register normal class members
        else:
            super(RuleAccu, self).__setitem__(k, v)


class cfgmeta(type):


    @classmethod
    def __prepare__(mcls, n, bs, **kw):
        return RuleAccu()
    

    def __new__(mcls, n, bs, accu):

        """
        Check declaration/usage information.
        Complete ordinal helpers. 
        """
        accu.lxs.move_to_end('IGNORED', last=False)
        accu.lxs.move_to_end('END')

        undeclared = []
        unused_lxs = set(accu.lxs)
        unused_lxs.discard('IGNORED')
        unused_lxs.discard('END')

        for j, r in enumerate(accu.rules): 
            for y in r.rhs:
                unused_lxs.discard(y)
                if y not in accu.lxs and y not in accu.ntms:
                    undeclared.append((y, j, r)) 

        if unused_lxs:
            msg = "\nUnused Lex-symbols: {}".format("\n".join(unused_lxs))
            print(msg)
            # warnings.warn(msg)

        if undeclared:
            msg = "\nUndeclared symbols: {}".format(
                "\n".join("'{}' in {}th rule {}".format(y, j, r)
                          for y, j, r in undeclared))
            raise ValueError(msg)

        else: 

            _top = '_top'
            if _top not in accu.ntms:
                # First rule to be closed by _top rule
                subtop = accu.rules[0].lhs
                top_rule = Rule(_top, (subtop, 'END'), lambda s, end: s)
                accu.rules.insert(0, top_rule)

            G = Grammar(accu.lxs, accu.ntms, accu.rules, _top)
            accu['parser'] = G
            return super().__new__(mcls, n, bs, accu)


