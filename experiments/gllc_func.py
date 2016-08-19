# GLL combinator
import re

from pprint import pprint

def idfunc(x):
    return x

def idseq(*x):
    return x


def token(tk, hdl=idfunc):
    if not tk:
        assert False, 'Empty token'
    def _t(inp):
        if inp and inp.startswith(tk):
            yield hdl(tk), inp[len(tk):]
    return _t

def rtoken(pat, hdl=idfunc):
    rgx = re.compile(pat)
    def _t(inp):
        m = rgx.match(inp)
        if m:
            yield hdl(m.group()), inp[m.end():]
    return _t


a, b, c = map(token, 'abc')
x, y, z = map(token, 'xyz')
res = list(a('abc'))
assert res == [('a', 'bc')]


def seq(*psrs, hdl=idseq):
    def _s(inp):
        agd = [((), inp)]
        for psr in psrs:
            agd1 = []
            while agd:
                rs, inp = agd.pop()
                for r, inp1 in psr(inp):
                    agd1.append((rs + (r,), inp1))
            agd = agd1
        for sq, inp1 in agd:
            yield hdl(*sq), inp1
    return _s

abc = seq(a, b, c)
assert list(abc('abc')) == [(('a', 'b', 'c'), '')], list(abc('abc'))
assert list(abc('abd')) == []


def alt(psrs):
    def _a(inp):
        for psr in psrs:
            yield from psr(inp)
    return _a

xyz = alt([x, y, z])
assert (list(xyz('x'))) == [('x', '')]
assert (list(xyz('y'))) == [('y', '')]
assert (list(xyz('z'))) == [('z', '')]
ab_ac = alt([seq(a, b),
             seq(a, c)])
assert (list(ab_ac('ab'))) == [(('a', 'b'), '')]
assert (list(ab_ac('ac'))) == [(('a', 'c'), '')]

# assert 0

def many(psr, hdl=idfunc):
    def _m(inp):
        # # rec version
        # has1 = 0
        # for r, inp1 in psr(inp):
        #     has1 = 1
        #     for rs, inp2 in _m(inp1):
        #         yield [r] + rs, inp2 
        # if not has1:
        #     yield [], inp

        # iter version
        agd = [([], inp)]
        while agd:
            agd1 = []
            for rs, inp in agd:
                for r, inp1 in psr(inp):
                    agd1.append((rs+[r], inp1))
            # Until no next match agd1.
            if not agd1:
                for rs, inp in agd:
                    yield hdl(rs), inp
            agd = agd1
    return _m

res = list(many(alt([a, b]))('aba'))
assert (res) == [(['a', 'b', 'a'], '')]


def opt(psr):
    def _o(inp):
        has1 = 0
        for r, inp1 in psr(inp):
            has1 = 1
            yield r, inp1
        if not has1:
            yield [], inp
    return _o


# pprint(list(alt([a, b])('ac')))
# res = list(opt(alt([a, b]))('ac'))
# pprint(res)
# res = list(opt(alt([a, b]))('c'))
# pprint(res)


white = many(alt([token(' '), token('\t'), token('\n'), token('\v')]))
def word(lit):
    def _w(inp):
        for (w, s), inp1 in seq(token(lit), white)(inp):
            yield w, inp1
    return _w

def rword(lit, hdl=idfunc):
    def _w(inp):
        for (w, s), inp1 in seq(rtoken(lit), white)(inp):
            yield hdl(w), inp1
    return _w

digit = rtoken('\d+\s*')

def p_int(inp):
    for n, inp1 in digit(inp):
        yield int(n), inp1

ints = many(p_int)
pprint(list(ints('1 23 456 ')))


def parser(func):
    def _p(inp):
        return func()(inp)
    return _p

@parser
def stmt():
    i = word('if')
    t = word('then')
    e = word('else')
    s = word('s')
    x = word('0')
    return alt([
        s,
        seq(i, x, t, stmt, e, stmt),
        seq(i, x, t, stmt),
    ])

pprint(list(stmt('s')))
pprint(list(stmt('if 0  then s else     s')))
pprint(list(stmt('if  0 then  s')))

@parser
def sexp():
    return alt([
        rword('\w+'),
        seq(word('('), many(sexp), word(')'),
            hdl=lambda *tp: tp[1])
    ])

sexp = parser(
    lambda: alt([
        rword('\w+'),
        seq(word('('), many(sexp), word(')'),
            hdl=lambda *tp: tp[1])
    ]))

pprint(list(sexp('abc de')))
pprint(list(sexp('( abc de  )')))
pprint(list(sexp('( abc de (f (g)) )')))


@parser
def p_expr():

    num = rword('\d+', hdl=int)
    plus = word('+')
    times = word('*')
    left = word('(')
    right = word(')')

    def expr(inp):
        return seq(term, many(seq(plus, term, hdl=lambda p, t: t),
                              hdl=sum),
                   hdl=lambda t, ts: t + ts)(inp)

    def term(inp):
        def product(xs):
            p = 1
            for x in xs: p *= x
            return p

        return seq(factor, many(seq(times, factor, hdl=lambda t, f: f),
                                hdl=product),
                   hdl=lambda f, r: f * r)(inp)

    factor = alt([
        num,
        seq(word('('), expr, word(')'), hdl=lambda s:s[1])
    ])

    return expr


print()
pprint([*p_expr('3 + 1 + 9')])
pprint([*p_expr('3 + 2 * 9 + 1')])


def p_json(inp):

    number = rword('\d+', hdl=int)
    
    @parser
    def object():
        return alt([
            seq(word('{'), word('}'), hdl=lambda l, r: {}),
            seq(word('{'), members, word('}'), hdl=lambda l, ms, r: dict(ms))
        ])

    @parser
    def members():
        return seq(pair, many(seq(word(','), pair, hdl=lambda w, p: p)),
                   hdl=lambda p, m: [p] + m)

    @parser
    def string():
        return seq(word("\""), rword('\w*'), word("\""),
                   hdl=lambda l, w, r: w)

    @parser
    def pair():
        return seq(string, word(':'), value,
                   hdl=lambda s, w, v: (s, v))

    @parser
    def array():
        return alt([
            seq(word('['), many(elements), word(']'),
                hdl=lambda l, m, r: m)
        ])

    @parser
    def elements():
        return seq(value,
                   many(seq(word(','), value, hdl=lambda c, v: v)),
                   hdl=lambda v, m: [v] + m)

    value = alt([
        string,
        number,
        object,
        array,
        word('true'),
        word('false'),
        word('null'),
    ])

    return object(inp)

pprint(list(p_json('{   }')))
pprint(list(p_json('{ "a": 123 , "b": 987 }')))
