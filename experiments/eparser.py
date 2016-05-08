from collections import namedtuple as data

FAIL = (None, None)

def unit(x):
    return (x,)

def mappend(a, b):
    return a + b

def msum(us):
    return sum((u for u in us), ())

def fails(a, b):
    return (a, b) == FAIL

is_a = isinstance

empty = data('empty', '')
EMPTY = empty()
atom = data('atom', 'val')
seq  = data('seq', 'fst snd')
alt  = data('alt', 'op1 op2')

# GROUP is a virtual object.
# - by constructing the parser syntax tree explicitly,
#   no grouping is implicitly constructed as a subtree. 
grp  = data('grp', 'val')
star = data('star', 'val')
opt  = data('opt', 'val')
nest = data('nest', 'val')

# PARSE-RESULT is represented as a tuple
# PARSE-RESULT :: ( <parsed-construct-monoid> * <rest-inp-string> )

def parse(x, inp):
    # Dynamic dispatch
    prs = eval('parse_{}'.format(type(x).__name__))
    return prs(x, inp)


def match_atom(x, inp):
    return inp.startswith(x.val)

def parse_atom(x, inp):
    if match_atom(x, inp):
        return (unit(x.val.upper()), inp[len(x.val):]) # CONS here.
    else:
        return FAIL

def parse_grp(x, inp):
    return parse(x, inp)

def parse_seq(seq, inp):
    y, rest1 = parse(seq.fst, inp)
    if not fails(y, rest1):
        z, rest2 = parse(seq.snd, rest1)
        if not fails(z, rest2):
            return (mappend(y, z), rest2) # Another CONS here
    return FAIL

def parse_alt(alts, inp):
    y, rest1 = parse(alts.op1, inp)
    if not fails(y, rest1):
        return y, rest1
    else:
        y, rest2 = parse(alts.op2, inp)
        if not fails(y, rest2):
            return y, rest2
        else:
            return FAIL

def parse_end(x, inp):
    if not inp:
        return FAIL
    else:
        return (x, '')

def parse_empty(x, inp):
    return (EMPTY, inp)         # UNIT, which can be hidden by semantical operations.

def parse_star(x, inp):         # Eager
    rep = []
    while inp:
        y, inp1 = parse(x.val, inp) # Try forehead.
        if fails(y, inp1):
            return (msum(rep), inp)
        else:
            rep.append(y)
            inp = inp1
    return (msum(rep), inp)

def parse_opt(x, inp):
    y, inp1 = parse(x.val, inp)
    if fails(y, inp1):
        return parse(EMPTY, inp)
    else:
        return (y, inp1)

# def parse_nest(x, inp, depth=0):
#     if x == EMPTY:
#         return (EMPTY, inp)

# eval with environment???

parse(atom('a'), 'a')
parse(seq(atom('a'), atom('b')), 'ab')
parse(seq(atom('a'), atom('b')), 'aB')
parse(seq(atom('a'), empty()), 'abb') # Each initial expression should be treated as seq.
parse(alt(atom('a'), atom('b')), 'a')
parse(alt(atom('a'), atom('b')), 'b')
parse(alt(atom('a'), atom('b')), 'c')
parse(star(atom('a')), '')
parse(star(atom('a')), 'b')
parse(star(atom('a')), 'a')
parse(star(atom('a')), 'aaaa')
parse(star(atom('a')), 'aaaab')
parse(star(seq(atom('a'), atom('b'))), '')
parse(star(seq(atom('a'), atom('b'))), 'ab')
parse(star(seq(atom('a'), atom('b'))), 'abc')
parse(star(seq(atom('a'), atom('b'))), 'abab')
parse(star(seq(atom('a'), atom('b'))), 'ababc')
parse(atom('a'), 'bc')
parse(opt(atom('a')), 'bc')
parse(opt(atom('a')), 'abc')
parse(seq(opt(atom('a')), atom('b')), 'abc')
parse(seq(opt(atom('a')), atom('b')), 'bbc')
parse(EMPTY, 'aa')

i1 = alt(seq(atom('a'), atom('b')),
         seq(atom('c'), atom('d')))

parse(i1, 'cd')
parse(i1, 'ab')
parse(i1, 'abc')
parse(i1, 'ac')
