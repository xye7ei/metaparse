import preamble
from metaparse import Symbol, Rule, Grammar, LALR
from pprint import pprint

class read(type):

    class gs(object):
        def __init__(self):
            self.lexes = []
            self.pats = []
            self.rules = []
            self.prece = {}
        def __setitem__(self, k, v):
            if not k.startswith('__'):
                # lexical tuple
                if isinstance(v, tuple):
                    assert len(v) == 2
                    l, p = v
                    self.lexes.append(k)
                    self.pats.append(l)
                    self.prece[k] = p
                # lexical str
                elif isinstance(v, str):
                    self.lexes.append(k)
                    self.pats.append(v)
                # alternatives
                elif isinstance(v, (list, set)):
                    for alt in v:
                        if not isinstance(alt, (list, tuple)):
                            alt = (alt,)
                        rhs = []
                        for x in alt:
                            if isinstance(x, Symbol):
                                rhs.append(str(x))
                            elif isinstance(x, str):
                                self.lexes.append(x)
                                self.pats.append(None)
                                rhs.append(x)
                        self.rules.append(Rule(k, rhs))
                # 
                elif callable(v):
                    pass
        def __getitem__(self, k0):
            return Symbol(k0)

    @classmethod
    def __prepare__(mcls, n, bs, **kw):
        return read.gs()
    def __new__(mcls, n, bs, gs):
        return Grammar(gs.lexes, gs.pats, gs.rules, prece=gs.prece)


class E(metaclass=read): 

    # IGNORED = r'\s+'

    NEG = r'!'   , 5
    CON = r'&'   , 4
    DIS = r'\|'  , 3
    IMP = r'->'  , 2
    IFF = r'<=>' , 1

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

# g = Grammar(*E)
# pprint(g)
pprint(E.lex2pats)
p = LALR(E)

# pprint([*p.lexer.tokenize('True & False', True)])
# pprint(p.parse('P & Q | R & !S'))

s = p.dump('meta2_dumps.py')
p1 = LALR.load('meta2_dumps.py', globals())

# print(s)

pprint(p1.parse('P & Q | R & !S'))
