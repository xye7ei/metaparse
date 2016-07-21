import preamble
import unittest

from metaparse import *


class G(metaclass=cfg):

    a, b, c, d = r'abcd'

    def S(A, B, C): pass
    def S(D): pass
    def A(a, A): pass
    def A(): pass
    def B(B, b): pass
    def B(): pass
    def C(c): pass
    def C(D): pass
    def D(d, D): pass
    def D(E): pass
    def E(D): pass
    def E(B): pass


class TestGrammar(unittest.TestCase):

    def test_first(self):
        self.assertEqual(G.FIRST['S'], {'a', 'd', EPSILON})

    def test_first_many(self):
        self.assertEqual(G.first_of_seq(['A', 'B', 'C'], '#'), {'a', 'd', 'c', 'd', '#'})

    def test_nullalbe(self):
        self.assertEqual(G.NULLABLE, {'S^', 'S', 'A', 'B', 'C', 'D', 'E'})

if __name__ == '__main__':
    unittest.main()
    # from pprint import pprint

    # pprint(G.PRED_TREE['S'])
    # pprint(G.FIRST['S'])
    # pprint(G.NULLABLE)
