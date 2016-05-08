from collections import namedtuple

def _id(x):
    return x


def token(tk, trans=_id):
    """
    :tk:
    Token symbol can be empty (trivial)
    """
    def _token(inp):
        """
        :inp: emptiness leads to fail
        """
        if not tk:
            return trans(tk), inp
        elif inp and inp.startswith(tk):
            return trans(tk), inp[len(tk):]
        else:
            return None, inp        # `inp` for backtracking
    return _token


def sequence(pars, trans):
    """
    :pars:
    A list of parsers as a sequence.
    :trans:
    Ideally, the arguments of `sequence` is a
    heterogenous sequence of translated results,
    among which there can be of different types.
    This argument cannot be ignored to have any
    default definition since no translator supplies
    universal product type!!
    """
    def _sequence(inp):
        args = []
        inpr = inp
        for par in pars:
            t, inpr = par(inpr)
            if t is None:
                return None, inp
            else:
                args.append(t)
        return trans(*args), inpr
    return _sequence


def many(par, trans=_id) -> ([object], str):
    """
    :->:
    A list when no translation is set
    :par:
    Parser to be replicatedly applied
    """
    def _many(inp):
        seq = []
        inpr = inp
        while 1:
            t, inpr = par(inpr)
            if t is None:
                break
            else:               # `inpr` remains unconsumed once
                seq.append(t)
        return trans(seq), inpr
    return _many


def many1(par, trans=_id):
    def _many1(inp):
        t, inpr = many(par)(inp)
        if t == []:
            return None, inp
        else:
            return trans(t), inpr
    return _many1


def alter(pars, trans=_id):
    """
    :->:
    The first successful parsed result
    :pars:
    A list of parsers
    :trans: 
    Maybe there is no need to translated any result
    which is already translated by one branch of the
    alternatives
    """
    def _alter(inp):
        for par in pars:
            t, inp = par(inp)
            if t is not None:
                return t, inp
        return None, inp
    return _alter


# def optional(wd, inp, trans=_id):
token('')('')
token('')('abc')
token('a')('abc')
many(token('a'))('aaac')
alter((token('a'), token('b')))('abc')
alter((token('a'), token('b')))('bbc')


def digit(inp: str) -> (int, str):
    alts = []
    for i in range(10):
        d = str(i)
        alts.append(token(d))
    d, inpr = alter(alts)(inp)
    if d is not None:
        return int(d), inpr
    else:
        return None, inp


def integer(inp: str) -> (int, str):
    dseq, inpr =  many(digit)(inp)
    if dseq:
        num = 0
        for d in dseq:
            num = num * 10 + d
        return num, inpr
    else:
        return None, inp
            
# def double(inp: str) -> (float, str):

whitespace = alter([
    token(' '),
    token('\t'),
    token('\n'),
    token('\r'),
])
whitespaces = many(whitespace, lambda s: ' ')
whitespaces('    \t   \r   \n\n ')
char = alter([token(x) for x in 'abcdefghijklmnopqrstuvwxyz'])
symbol = many1(char, lambda cs: ''.join(cs))


def tailwhite(par):
    return sequence([par, many(whitespace)],
                    lambda ctnt, w: ctnt)

empty = token('')
word = tailwhite(symbol)

def parens(inp: str) -> (int, str):
    return alter([
        sequence([token('('),
                  many(whitespace),
                  parens,
                  many(whitespace),
                  token(')'), 
              # ], lambda l, w, p, y, r: p + 1),
              ], lambda *a: a[2] + 1),
        token('', trans=lambda x: 0),
    ])(inp)


def sexp(inp: str) -> (list, str):
    alt1 = word
    alt2 = sequence([
        token('('),
        many(whitespace),
        many(sexp),
        token(')'),
        many(whitespace),
    ], lambda l, w1, e, r, w2: e)
    return alter([alt1, alt2])(inp)

if __name__ == '__main__':

    parens('(((  )  ))')
    
    many1(token('a'))('bcd')
    many1(token('a'), lambda xs: len(xs))('aaaaabcd')
    integer('1234abc')
    symbol('abc  \t\n\r')
    word('abc  \t\n\r')
    symbol('  abc  \t\n\r')

    sexp('abc')
    sexp('()')
    sexp('(foo (bar baz) fii)')

    # Rule representation and semantics
    # LEFT -> (           ; \x -> x  :: T_left
    # RIGHT -> )          ; \x -> x :: T_right
    # S -> \w+            ; \x -> x :: T_s
    # S -> LEFT S RIGHT   ; \l, s, r -> s ::
    #                         T_left * T_s * T_right -> T_s
    # Recursive type raised...
    # + least-fixed-point of recursion


