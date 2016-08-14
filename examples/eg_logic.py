import preamble

from metaparse import Symbol, LALR
from collections import namedtuple

NOT = Symbol('NOT')
AND = Symbol('AND')
OR = Symbol('OR')
IMP = Symbol('IMP')
IFF = Symbol('IFF')

class PropLogic(metaclass=LALR.meta):

    T = r'True'
    F = r'False'
    W = r'[A-Z]\w*'

    L = r'\(' ; R = r'\)'
    LL = r'\['; RR = r'\]'

    NEG = r'!'   , 5
    CON = r'&'   , 4
    DIS = r'\|'  , 3
    IMP = r'->'  , 2
    IFF = r'<=>' , 1

    def Sentence(Atomic):
        return Atomic
    def Sentence(Complex):
        return Complex

    def Atomic(T):
        return True
    def Atomic(F):
        return False
    def Atomic(W):
        return table[W]

    def Complex(L, Sentence, R):
        return Sentence
    def Complex(LL, Sentence, RR):
        return Sentence
    def Complex(NEG, Sentence):
        return not Sentence
    def Complex(Sentence, CON, Sentence_1):
        return Sentence and Sentence_1
    def Complex(Sentence, DIS, Sentence_1):
        return Sentence or Sentence_1
    def Complex(Sentence, IMP, Sentence_1):
        return not Sentence or Sentence_1
    def Complex(Sentence, IFF, Sentence_1):
        return Sentence == Sentence_1


inp = """
(P & Q | R & !S)
"""

table = dict(
    P=True,
    Q=False,
    R=True,
    S=False,
)

t = PropLogic.parse(inp)
r = t.translate()

from pprint import pprint

pprint(t)
pprint(r)

# pprint(PropLogic.__dict__)
