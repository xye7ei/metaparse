from eg_dumps_file import *

import re
import types
import marshal
import warnings
import pprint as pp

from collections import namedtuple, deque

lex2rgxs = [(lex, re.compile(pat)) for lex, pat in lex2pats]

lex_handlers = {
    name: types.FunctionType(marshal.loads(src), globals())
    for name, src in lex_handler_sources.items()
}

semans = [
    types.FunctionType(marshal.loads(src), globals())
    for src in seman_sources
]

Rule = namedtuple('Rule', 'lhs rhs')
Rule.__repr__ = lambda s: '({} = {})'.format(s.lhs, ' '.join(s.rhs))

Item = namedtuple('Item', 'rule pos')
Item.__repr__ = lambda s: '({} = {}.{})'.format(s.rule.lhs,
                                                ' '.join(s.rule.rhs[:s.pos]),
                                                ' '.join(s.rule.rhs[s.pos:]))

rules = [Rule(l, r) for l, r in rules]
Ks = [[Item(rules[r], pos) for r, pos in K] for K in Ks]


Token = namedtuple('Token', 'at symbol lexeme value')
Token.__repr__ = lambda s: '({} = {})'.format(s.symbol, repr(s.value))



def tokenize(inp, with_end=True):

    pos = 0
    while pos < len(inp):
        # raw string match
        raw_match = False
        # re match
        n = None
        m = None
        for cat, rgx in lex2rgxs:
            # raw
            if rgx is None:
                if inp.startswith(cat, pos):
                    yield Token(pos, cat, cat, cat)
                    pos += len(cat)
                    raw_match = True
                    break
            # re
            else:
                m = rgx.match(inp, pos=pos)
                # The first match with non-zero length is yielded.
                if m and len(m.group()) > 0:
                    n = cat
                    break
        if raw_match:
            continue
        elif m:
            assert isinstance(n, str)
            if n == 'IGNORED':
                # Need IGNORED handler?
                at, pos = m.span()
            elif n == 'ERROR':
                # Call ERROR handler!
                at, pos = m.span()
                lxm = m.group()
                if 'ERROR' in lex_handlers:
                    # Suppress error token and call handler.
                    lex_handlers[ERROR](lxm)
                    # yield Token(at, ERROR, lxm, h(lxm))
                else:
                    # Yield error token when no handler available.
                    yield Token(at, ERROR, lxm, lxm)
            else:
                at, pos = m.span()
                lxm = m.group()
                if n in lex_handlers:
                    # Call normal token handler.
                    h = lex_handlers[n]
                    # Bind semantic value.
                    yield Token(at, n, lxm, h(lxm))
                else:
                    yield Token(at, n, lxm, lxm)
        else:
            # Report unrecognized Token here!
            msg = '\n'.join([
                '',
                '=========================',
                'No defined pattern starts with char `{}` @{}'.format(inp[pos], pos),
                '',
                '* Consumed input: ',
                repr(inp[:pos]),
                '=========================',
                '',
            ])
            raise GrammarError(msg)
    if with_end:
        yield Token(pos, 'END', None, None)


def parse(inp, interp=False, n_warns=5):

    trees = []
    sstack = [0]

    toker = tokenize(inp, with_end=True) # Use END to force finishing by ACCEPT
    tok = next(toker)
    warns = []

    try:
        while 1:

            # Peek state
            s = sstack[-1]

            if tok.symbol not in ACTION[s]:
                msg = '\n'.join([
                    '',
                    'WARNING: ',
                    'LALR - Ignoring syntax error reading Token {}'.format(tok),
                    '- Current kernel derivation stack:',
                    pp.pformat([Ks[i] for i in sstack]),
                    '- Expecting tokens and actions:',
                    pp.pformat(ACTION[s]),
                    '- But got: \n{}'.format(tok),
                    '',
                ])
                warnings.warn(msg)
                warns.append(msg)
                if len(warns) == n_warns:
                    raise ValueError(
                        'Warning tolerance {} reached. Parsing exited.'.format(n_warns))
                else:
                    tok = next(toker)

            else:
                act, arg = ACTION[s][tok.symbol]

                # SHIFT
                if act == 'SHIFT':
                    if interp:
                        trees.append(tok.value)
                    else:
                        trees.append(tok)
                    sstack.append(GOTO[s][tok.symbol])
                    # Go on scanning
                    tok = next(toker)

                # REDUCE
                elif act == 'REDUCE':
                    assert isinstance(arg, int)
                    rule = lhs, rhs = rules[arg]
                    seman = semans[arg]
                    subts = deque()
                    for _ in rhs:
                        subt = trees.pop()
                        subts.appendleft(subt)
                        sstack.pop()
                    if interp:
                        tree = seman(*subts)
                    else:
                        tree = ((rule, seman), list(subts))
                    trees.append(tree)
                    sstack.append(GOTO[sstack[-1]][lhs])

                # ACCEPT
                elif act == 'ACCEPT':
                    # Reduce the top semantics.
                    assert isinstance(arg, int), arg
                    rule = rules[arg]
                    seman = semans[arg]
                    if interp:
                        return seman(*trees)
                    else:
                        assert len(trees) == 1
                        return trees[0]
                else:
                    raise ValueError('Invalid action {} on {}'.format(act, arg))

    except StopIteration:
        raise ValueError('No enough tokens for completing the parse. ')


def interpret(inp):
    return parse(inp, interp=True)

table = {}

inp = 'x = 1 + 2 7 ** 3 * 5 + 9'

ts = list(tokenize(inp))
pp.pprint(ts)

r = interpret(inp)

pp.pprint(table)
pp.pprint(r)
