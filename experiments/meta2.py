import preamble
from metaparse import Symbol, Rule, Grammar, LALR
from pprint import pprint

class read(type):

    class gs(list):
        def __init__(self):
            super(read.gs, self).__init__()
            self.lex2pats = []
            self.rules = []
            self.prece = {}
        def __setitem__(self, k, v):
            if not k.startswith('__'):
                if isinstance(v, tuple):
                    assert len(v) == 2
                    l, p = v
                    self.lex2pats.append((k, l))
                    self.prece[k] = p
                elif isinstance(v, str):
                    self.lex2pats.append((k, v))
                elif isinstance(v, (list, set)):
                    for alt in v:
                        if not isinstance(alt, (list, tuple)):
                            alt = (alt,)
                        rhs = []
                        for x in alt:
                            if isinstance(x, Symbol):
                                rhs.append(str(x))
                            elif isinstance(x, str):
                                self.lex2pats.append((x, None))
                                rhs.append(x)
                        self.rules.append(Rule(k, rhs))
        def __getitem__(self, k0):
            return Symbol(k0)

    @classmethod
    def __prepare__(mcls, n, bs, **kw):
        return read.gs()
    def __new__(mcls, n, bs, gs):
        return gs.lex2pats, [(rl, None) for rl in gs.rules], gs.prece


class E(metaclass=read): 

    IGNORED = r'\s+'

    NEG = r'!'   , 5
    CON = r'&'   , 4
    DIS = r'\|'  , 3
    IMP = r'->'  , 2
    IFF = r'<=>' , 1

    # L, R = r'\(', '\)'

    W   = r'[A-Z]\w*'

    Sentence = [
        Atomic,
        Complex,
    ]

    Atomic = [
        'True',
        'False',
        W,
    ]

    Complex = [
        ['(', Sentence, ')'],
        ['[', Sentence, ']'],
        [NEG, Sentence],
        [Sentence, CON, Sentence],
        [Sentence, DIS, Sentence],
        [Sentence, IMP, Sentence],
        [Sentence, IFF, Sentence],
    ]


# pprint(E)

g = Grammar(*E)
# pprint(g)
pprint(g.lex2pats)
p = LALR(g)

# pprint([*p.lexer.tokenize('True & False', True)])
pprint(p.parse('P & Q | R & !S'))
