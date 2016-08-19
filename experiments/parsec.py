"""This module contains experimental fundamental Parsec-like utilities
with Python style (iterative) implementations.

"""
from collections import namedtuple


# Structures

L = Left = namedtuple('Left', 'info')
L.__add__ = lambda l1, l2: Left('{}\n{}'.format(l1, l2))

R = Right = namedtuple('Right', 'value')

isL = lambda x: isinstance(x, Left)
isR = lambda x: isinstance(x, Right)


# Core combinators

def _id(x):
    return x


def token(tk, trans1=_id):
    if not tk:
        raise ValueError('Empty token is not allowed!')
    def _t(inp):
        if not inp:
            return L('No input.'), inp
        elif inp.startswith(tk):
            return R(trans1(tk)), inp[len(tk):]
        else:
            # return L('Wrong token: expecting {}'.format(tk)), inp
            return L('Wrong token: expecting {}'.format(tk)), inp
    return _t


def seq(ps, trans_tuple=_id):
    def _s(inp):
        args = ()               # tuple for heterogenous elements in sequence
        inpr = inp
        for p in ps:
            res, inpr = p(inpr)
            if isL(res):
                return L('seq failed: {}'.format(p.__name__, repr(inp))) + res, inp
            else:
                args += (res.value,)
        return R(trans_tuple(args)), inpr
    return _s


def many(par, trans_list=_id):
    def _m(inp):
        lst, inpr = [], inp
        while 1:
            res, inpr = par(inpr)
            if isL(res):
                break           # ignore error
            else:
                lst.append(res.value)
        return R(trans_list(lst)), inpr
    return _m


def many1(par, trans_list=_id):
    def _m1(inp):
        res, inpr = par(inp)
        if isL(res):
            return res, inp
        else:
            lst, inpr = [res.value], inpr
            while 1:
                res, inpr = par(inpr)
                if isL(res):
                    break
                else:
                    lst.append(res.value)
            return R(trans_list(lst)), inpr
    return _m1


def alt(pars):
    def _a(inp):
        for par in pars:
            res, inpr = par(inp) # full backtracking with :inp:
            if isR(res):
                return res, inpr
        return L('No matching alternatives.'), inp
    return _a


# Test core unitilities

assert token('a')('abc') == (R('a'), 'bc')
assert many(token('a'))('aaac') == (R(['a', 'a', 'a']), 'c')
assert alt([token('a'), token('b')])('abc') == (R('a'), 'bc')
assert alt((token('a'), token('b')))('bbc') == (R('b'), 'bc')
assert alt([seq([*map(token, 'ab')]),
            seq([*map(token, 'ac')])])('ac') == (R(('a', 'c')), '')
assert alt([])('') == (L(info='No matching alternatives.'), '')


# Extended basic utilities
        
# Single whitespace
whitespace = alt([
    token(' '),
    token('\t'),
    token('\n'),
    token('\r'),
])
# Multiple whitespaces
whitespaces = many(whitespace, lambda s: ' ')

assert whitespaces('') == (R(' '), ''), whitespaces('')
assert whitespaces('    \t   \r   \n\n ') == (R(' '), '')

char = alt([token(x) for x in 'abcdefghijklmnopqrstuvwxyz'])
symbol = many1(char, lambda cs: ''.join(cs))

assert symbol('abcd efg') == (Right('abcd'), ' efg')

def trailed(par):
    return seq((par, many(whitespace)), lambda s: s[0])

def tokenw(lit):
    return seq([token(lit), whitespaces], lambda sw: sw[0])

print(tokenw('abcd')('abcd   cfg'))
assert tokenw('xy')('xy') == (R('xy'), ''), tokenw('xy')('xy')
assert tokenw('abcd')('abcd   efg') == (R('abcd'), 'efg'), tokenw('abcd')('abcd   efg')


# Exemplar usages for practical structure

digit = alt([token(str(i), lambda x: ord(x) - ord('0')) for i in range(10)])

def integer(inp):
    dlst, inpr =  many(digit)(inp)
    if isR(dlst):
        num = 0
        for d in dlst.value:
            num = num * 10 + d
        return R(num), inpr
    else:
        return L('Not a integer.'), inp


assert digit('159') == (Right(1), '59')
assert digit('9') == (Right(9), '')
assert integer('12345 67') == (Right(12345), ' 67')

pa_l1 = trailed(token('('))
pa_r1 = trailed(token(')')) 
def parens(inp: str) -> (int, str):
    return alt([
        seq([tokenw('('),
                  parens,
                  tokenw(')'),
              ], lambda seq: seq[1] + 1),
              # ]),
        seq([], lambda x: 0)
        # seq([]),
    ])(inp)

assert parens('   ') == (Right(0), '   ')
assert parens('( )') == (Right(1), '')
assert parens('((( (  ) )  ))') == (Right(4), '')

def sexp(inp):
    return alt([
        seq([symbol,
             whitespaces], lambda s: s[0]),
        seq([tokenw('('),
             many(sexp),
             tokenw(')')], lambda s: s[1])
    ])(inp)

assert sexp('a') == (R('a'), '')
assert sexp('()') == (R([]), '')
assert sexp('(  b b   ops )') == (R(['b', 'b', 'ops']), ''), sexp('(  b b   ops )')
assert sexp('( a (b (c d)) ((e)))') == (R(['a', ['b', ['c', 'd']], [['e']]]), '')

# Rule representation and semantics
# LEFT -> (           ; \x -> x  :: T_left
# RIGHT -> )          ; \x -> x :: T_right
# S -> \w+            ; \x -> x :: T_s
# S -> LEFT S RIGHT   ; \l, s, r -> s ::
#                         T_left * T_s * T_right -> T_s
# Recursive type raised...
# + least-fixed-point of recursion


# Left-factoring
# Let |a| == len(inp) 
# |a| == 2: S(|a|) == (2, 0)
# |a| == 3: S(|a|) == (2, 1)
# |a| == 4: S(|a|) == (4, 0)
# |a| == 5: S(|a|) == (2, 3), since FAIL with a(.Sa=> aaaax) $
#                             thus  SUCC with a(.a=> a) aaa$
# |a| == 6: S(|a|) == (4, 2), since FAIL with aa(.Sa=> aaaax)$
#                             and   SUCC with aa(.a=> a) a$
#                             and   SUCC with a(.S=> aa)a $
#                             and   SUCC with (S=> aSa) $
#     *. can be treated like
#       - The rightmost 5 "a"s yields (2, 3)
#       - The leftmost "a" comes to rescue, in cooperation with right 1
#       - The 2 "a"s recognized as "S" can be augmented with
#         left more 1 and right more 1 giving "aSa", a successful "S",
#         leaving still 2 on the right.
# |a| == 7: S(|a|) == (6, 1) 
# |a| == 8: S(|a|) == (8, 0) 
# |a| == 9: S(|a|) == (2, 7) 

aSa = lambda inp: alt([
    seq([tokenw('a'),
         aSa,
         tokenw('a')]),
    seq([tokenw('a'),
         tokenw('a')]),
])(inp)

print(aSa(' a'))
print(aSa('a  a'))
print(aSa('a a  a'))
print(aSa('a aa  a'))
print(aSa('a aa  a  a'))
print(aSa('a a a  a a  a'))
print(aSa('a a     a a   a a  a'))
print(aSa('a a a a  a   a a  a'))
