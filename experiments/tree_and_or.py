import re
import pprint as pp

from collections import namedtuple
from copy import deepcopy


# A canonical way of performing pure top-down parsing
# always produces OR-AND-Trees.

# - Each terminal produces a Leaf;
# - Each rule produces an AND-Tree;
# - Each non-terminal produces an OR-Tree;
#   - An OR-Tree should be associated with corresponded rest
#   input to allow full-backtracking! 

Leaf = namedtuple('Leaf', 'val')

ORTree = namedtuple('ORTree', 'nonterminal alts') # (str, [ANDTree])

ANDTree = namedtuple('ANDTree', 'seq rest') # ([ORTree | Leaf], rest)

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


def parse_and(G, rule, inp):
    seq = [] 
    live = []
    for Y in rule:
        if Y in G.nonterminals:
            t_or = parse_or(G, Y, inp)



g1 = Grammar([
    ('E', ['E', r'\+', 'T']),
    ('E', ['T']),
    ('T', ['T', r'\*', 'F']),
    ('T', ['F']),
    ('F', [r'\d+']),
    ('F', [r'\(', 'E', r'\)']),
])
