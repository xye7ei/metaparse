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

    NEG = r'!', 5
    CON = r'&', 4
    DIS = r'\|', 3
    IMP = r'->', 2
    IFF = r'<=>', 1

    def Sentence(Atomic):
        return Atomic
    def Sentence(Complex):
        return Complex

    def Atomic(T):
        return T
    def Atomic(F):
        return F
    def Atomic(W):
        return W

    def Complex(L, Sentence, R):
        return Sentence
    def Complex(LL, Sentence, RR):
        return Sentence
    def Complex(NEG, Sentence):
        return (NOT, Sentence)
    def Complex(Sentence, CON, Sentence_1):
        return (AND, Sentence, Sentence_1)
    def Complex(Sentence, DIS, Sentence_1):
        return (OR, Sentence, Sentence_1)
    def Complex(Sentence, IMP, Sentence_1):
        return (IMP, Sentence, Sentence_1)
    def Complex(Sentence, IFF, Sentence_1):
        return (IFF, Sentence, Sentence_1)


inp = """
(P & Q | R & !S)
"""

t = PropLogic.parse(inp)
r = t.translate()

from pprint import pprint

pprint(t)
pprint(r)

# pprint(PropLogic.__dict__)
