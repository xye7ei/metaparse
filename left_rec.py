# How to perform memoization for left-recursion? 

import re
import pprint as pp

from collections import namedtuple
from copy import deepcopy

Rule = namedtuple('Rule', 'lhs rhs')

class Grammar:

    def __init__(G, rules):
        G.nonterminals = set(l for l, _ in rules)
        G.terminals = set(r for _, rs in rules
                          for r in rs if r
                          if r not in G.nonterminals)
        G.symbols = G.nonterminals | G.terminals
        G.start_symbol = rules[0][0]
        G.rules = [Rule(l, r) for l, r in rules]

    def __repr__(G):
        return 'Grammar:\n{};\n'.format(pp.pformat(G.rules))


# The following algorithm applies only to LL(1) grammar with
# capability to handle left recursion using memoization.

# Memoized element is a pointer to Rule object paired with an index
# position in input, meaning which rule at which input position should
# not be used for parsing any more.

def parse_symbol(G, inp, X, k=0, memo=[]) -> (tuple, str):

    if X in G.terminals: 
        m = re.compile(X).match(inp, pos=k)
        if m: 
            return m.group(), m.span()[-1]
        else:
            return None, k 

    elif X in G.nonterminals:
        for rule in G.rules:
            if (rule, k) not in memo:
                if rule[0] == X:
                    tr, k1 = parse_rule(G, inp, rule, k, memo + [(rule, k)])
                    if tr:
                        return tr, k1
        return None, k

    else:
        raise ValueError('Invalid symbol {} in {}.'.format(X, G))


def parse_rule(G, inp, rule, k, memo) -> (tuple, str):

    n, rs = rule
    subs = []
    k1 = k

    for y in rs:
        sub, k1 = parse_symbol(G, inp, y, k1, memo)
        if sub:
            subs.append(sub)
        else:
            return None, k

    return (n, subs), k1


def parse(G, inp):

    tr, kend = parse_symbol(G, inp, G.start_symbol)
    if kend < len(inp):
        print('Pre-matured parse ending with rest {}.'.format(repr(inp[kend:])))

    return tr


# This grammar is ill for LL parsing not only because of left
# recursion, but also because of left sharing, which means left
# factoring is necessary!

# Otherwise general LL parer algorithm should be applied, which keeps
# tracks of every branching of alternatives, i.e. non-deterministics
# to allow complete backtracking, which is not possible in LL(1)
# framework. Then it is necessary to explictly maintain a stack.

# Algorithm: 
def _parse(G, inp):

    # stk :: [(<rule-id>, [<subtrees>], <input-index>)] 
    Leaf = namedtuple('Leaf', 'val i')
    Tree = namedtuple('Tree', 'rule subs i')
    Tree.ended = lambda t: len(t.rule.rhs) == len(t.subs)
    Tree.active = lambda t: t.rule.rhs[len(t.subs)]

    stks = []
    stk = [Tree(G.top_rule, [], 0)]

    while stks:
        stk = stks.pop()
        tr = stk.pop()
        if isinstance(tr, Tree):
            # COMPLETE
            if tr.ended():
                # ACCEPT
                if not stk:
                    # Check full parse?
                    return tr
                # SHIFT COMPLETED TREE
                else:
                    stk[-1].subs.append(tr)
            # EXPAND
            else:
                X = tr.active()
                # PUSH NEW TREE
                if X in G.nonterminals:
                    # Handle alternatives.
                    # Need forking!
                    for rule in G.rules:
                        stk1 = deepcopy(stk)
                        if rule.lhs == X:
                            sub = Tree(rule, [], i)
                            stk1.append(sub)
                            stks.append(stk1)
                # PUSH TERMINAL
                else:
                    m = re.compile(X).match(inp, pos=tr.i)
                    if m:
                        # Should Tree be immutable??
                        tr1 = Tree(tr.rule, tr.subs + [m.group], m.span()[-1])
                        stk.append(tr1)
                        stks.append(stk)
                    

g1 = Grammar([
    ('E', ['E', r'\+', 'T']),
    ('E', ['T']),
    ('T', ['T', r'\*', 'F']),
    ('T', ['F']),
    ('F', [r'\d+']),
    ('F', [r'\(', 'E', r'\)']),
])

g2 = Grammar([
    ('S', [r'\(', 'S', r'\)', 'S']),
    ('S', [])
])

g3 = Grammar([
    ('S', ['A', 's']),
    ('S', ['e']),
    ('A', ['S', 'a']),
])
parse(g3, 'easas')

# parse_symbol(g2, r'\(', '(())')
# parse_symbol(g2, r'S', '(())')

inp1 = '12+5*87'
tr1 = parse(g1, inp1)
tr1

inp2 = '(12+5)*87+3*(3+3)+3'
# parse(g1, inp2)
tr2 = parse(g1, inp2)
tr2

inp3 = '+'.join([inp2] * 100)

