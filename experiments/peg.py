# Experimental implementation for Parser Expression Grammar,
# represented by EBNF-like notation.

import re

from collections import namedtuple as data

Rule = data('Rule', 'lhs rhs')

# Expression is a superclass, which can be subclassed into
# - Terminal
# - Nonterminal
# - Alternatves
# - Sequence
# - Repeated/Star
# - Optional/Opt

Expr = data('Symbol', 'symbol')

Nonterminal = data('Nonterminal', 'symb')
Nonterminal = str
Terminal = data('Terminal', 'symb regexp')
Star = data('Star', 'sub')
Opt  = data('Opt', 'sub')
Plus = data('Plus', 'sub')
Seq  = data('Seq', 'subs')      # Using python list rather than CONS structure.
Alt  = data('Alt', 'subs')      # Using python list rather than CONS structure.
Nil  = None

is_a = isinstance

# Notes:

# To allow parsing expressions to include Sequence and Kleene Closure,
# there must be a corresponding sequenctial structure behaving as a
# primitive construction of a parse tree's subtrees. Theoretical it
# can be described as a monoid, which defines Unit(the empty) and
# Append(operation of accumulating).

# To represent the parsing result more simply, a parse result is
# either a ([Tree], Inp) or a (Tree, inp), whereas the latter`s first
# component can be represented as a singleton list.

# data Result = ([Tree], String) | (Tree, String) | FAIL

FAIL = (None, None)

def parse(G, x, inp):
    if is_a(x, Terminal):
        return parse_terminal(G, x, inp)
    elif is_a(x, Nonterminal):
        sub, inp1 = parse(G, G[x], inp)
        if (sub, inp1) == FAIL:
            return FAIL
        else:
            # Make a parse tree of 1 Nonterminal.
            # return (x.symb, sub), inp1
            return (x, sub), inp1
    elif is_a(x, Alt):
        return parse_alts(G, x.subs, inp)
    elif is_a(x, Seq):
        return parse_seq(G, x.subs, inp)
    elif is_a(x, Star):
        return parse_star(G, x.sub, inp)
    elif is_a(x, Opt):
        return parse_opt(G, x.sub, inp)
    else:
        raise TypeError('{} is not an expression.'.format(x))

def parse_terminal(G, x: Terminal, inp: str):
    if not inp:
        return FAIL
    else:
        m = re.match(x.regexp, inp, re.MULTILINE) # Matching MULTILINE activated.
        if not m:
            return FAIL
        else:
            _, end = m.span()
            tokval = re.sub(r'\s+', '', inp[:end])
            return (x.symb, tokval), inp[end:]

def parse_alts(G, alts: [Expr], inp: str) -> (tuple, str):
    """May return a OR-tree here. Recall each parse tree is an AND-OR
    tree.

    """
    pf = []
    for a in alts:
        t, inp1 = parse(G, a, inp)
        if (t, inp1) != FAIL:
            return (t, inp1)
    return FAIL

def parse_seq(G, subs: [Expr], inp: str) -> (tuple, str):
    ss = []
    for sub in subs:
        (t1, inp1) = parse(G, sub, inp)
        if (t1, inp1) != FAIL:
            # See whether the result is list or atom.
            if isinstance(t1, list):
                # For parse_star, parse_opt, parse_seq the result is a
                # list.
                ss.extend(t1)
            else:
                # For parse_terminal the result is an atom. It is a
                # singleton list as parse forest per se!!! For parse,
                # parse_alts the result maybe either.
                ss.append(t1)
            inp = inp1
        else:
            return FAIL
    # May convert singleton list to single node.
    if len(ss) == 1:
        return ss[0], inp
    else:
        return ss, inp


# Extended monoidic expressional structures.

def parse_star(G, sub: Expr, inp: str) -> (tuple, str):
    'sub is the expression enclosed by Star.'
    rep = []
    while 1 and inp:
        t1, inp1 = parse(G, sub, inp)
        if (t1, inp1) != FAIL:
            rep.append(t1)
            inp = inp1
        else:
            break
    return rep, inp

def parse_opt(G, sub: Expr, inp: str) -> (tuple, str):
    opt = []
    t1, inp1 = parse(G, sub, inp)
    if (t1, inp1) != FAIL:
        opt.append(t1)
        inp = inp1
    return [], inp


# i1 = Seq([Terminal('NUM', r'\d+'), Terminal('SPC', r'\s+'), Terminal('NUM', r'\d+')])
# i2 = Seq([Terminal('NUM', r'\d+'), Terminal('SPC', r'\s+'), Terminal('NUM', r'[A-Za-z_]\w+')])
# parse_seq(None, i1.subs, '123 456')
# parse_seq(None, i2.subs, '123 456')

# parse(None, i1, '123 456')
# parse(None, i2, '123 456') 
G1 = {Nonterminal('E'): Seq([Nonterminal('T'),
                             Star(Seq([Terminal('PLUS', r'\+'), Nonterminal('T')]))]),
      Nonterminal('T'): Seq([Nonterminal('F'),
                             Star(Seq([Terminal('TIMES', r'\*'), Nonterminal('F')]))]), 
      Nonterminal('F'): Terminal('NUM', r'\d+'),
}


# Bootstrapping grammar.
SPCS = r'\s*'

p_QUAL   = r'[\?\*\+]'

p_HEAD   = r'^'        + SPCS
p_LEFT   = r'\('       + SPCS
p_RIGHT  = r'\)'       + SPCS
p_SEMI   = r';'        + SPCS
p_ALT1   = r'/'        + SPCS
p_ALT2   = r'\|'       + SPCS
p_ALT    = r'[/\|]'    + SPCS
p_ARROW  = r'(->|::=)' + SPCS 
p_SYMBOL = r'[^;/\(\)\|\?\*\+\s]+' + SPCS

p_RIGHTQ = p_RIGHT + p_QUAL + r'?' + SPCS
p_SYMBOLQ= p_SYMBOL + p_QUAL + r'?' + SPCS

t_HEAD   = Terminal("HEAD"  , p_HEAD)
t_LEFT   = Terminal("LEFT"  , p_LEFT)
t_RIGHT  = Terminal("RIGHT" , p_RIGHT)
t_QUAL   = Terminal("QUAL"  , p_QUAL)
t_SEMI   = Terminal("SEMI"  , p_SEMI)
t_ALT1   = Terminal("ALT1"  , p_ALT1)
t_ALT2   = Terminal("ALT2"  , p_ALT2)
t_ALT    = Terminal("ALT"   , p_ALT)
t_ARROW  = Terminal("ARROW" , p_ARROW)
t_SYMBOL = Terminal("SYMBOL", p_SYMBOL)
t_RIGHTQ = Terminal("RIGHTQ", p_RIGHTQ)
t_SYMBOLQ= Terminal("SYMBOLQ", p_SYMBOLQ)

EBNF = {
    'Rules': Star('Rule'),
    'Rule': Seq(['LHS', t_ARROW, 'RHS']),
    'LHS': t_SYMBOL,
    'RHS': Seq(['Sequence',
                Star(Seq([t_ALT, 'Sequence'])),
                t_SEMI]),
    'Sequence': Star('Expr'),
    'Expr': Alt([t_SYMBOLQ,
                 Seq([t_LEFT, 'Sequence', t_RIGHTQ])]),
}

parse(EBNF, t_SYMBOL, 'ab')
parse(EBNF, t_SYMBOL, 'ab*')
parse(EBNF, t_SYMBOLQ, 'ab')
parse(EBNF, t_SYMBOLQ, 'ab*')
parse(EBNF, t_SYMBOLQ, 'ab  +')
parse(EBNF, ('Expr'), 'ab*')
parse(EBNF, ('Expr'), 'ab')
parse(EBNF, ('Expr'), 'ab*')
parse(EBNF, ('Expr'), 'ab  +')
parse(EBNF, ('Expr'), 'ab +;')
parse(EBNF, ('Expr'), "(plus E)")
parse(EBNF, ('Sequence'), 'ab + bc?;')
parse(EBNF, ('RHS'), "T (+ E) ;") # Error, using preserved symbol '+'
parse(EBNF, ('RHS'), "T (\+ E) ;")
parse(EBNF, ('RHS'), "T (plus E) ;")
parse(EBNF, ('RHS'), """T (plus E) | T; """)
parse(EBNF, ('RHS'), "a \+ b | a \* b | a? - b; ")
parse(EBNF, ('RHS'), "T plus E;")
parse(EBNF, ('RHS'), "T (plus E) ;")
parse(EBNF, ('Rule'), "<expr> -> a \+ b | a \* b | a? - b; ")
parse(EBNF, ('Rule'), "E -> T (plus T);")
res = parse(EBNF, Nonterminal('Rules'), """E -> T (plus T)*;
T -> F (times F)*;
F -> id;""")
res = parse(EBNF, Nonterminal('Rules'), """Expr -> atom | left Expr* right;
atom -> id;
""")


# Further functionalities:

# - Detect left factor: Test whether two alternatives of a rule share
# an identical FIRST token.

# - Detect left recursion: Test whether cycle exits after exploring
# derivation path.
if __name__ == '__main__':
    import pprint as pp
    pp.pprint(res)
