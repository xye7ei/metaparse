"""This file tests the fundamental checking mechanism of the class
Grammar.meta."""

import preamble
import warnings
import unittest

from metaparse import *

w = []

# with warnings.catch_warnings(record=True) as ws:
if 1:

    class G(metaclass=Grammar.meta):

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

    # assert len(ws) == 1

# pprint.pprint(G.terminals)
# pprint.pprint(G.nonterminals)


class TestGrammar(unittest.TestCase):

    def test_first_all(self):
        self.assertEqual(G.first_of_seq(['A', 'B', 'C'], '#'), {'a', 'b', 'c', 'd', '#'})

    def test_nullalbe(self):
        self.assertEqual(set(G.NULLABLE), {'S', 'A', 'B', 'C', 'D', 'E'})

    # def test_warn_loop(self):
    #     with warnings.catch_warnings(record=True) as ws:
    #         # Same as `G` above.
    #         class F(metaclass=cfg):
    #             a, b, c, d = r'abcd'
    #             def S(A, B, C): pass
    #             def S(D): pass
    #             def A(a, A): pass
    #             def A(): pass
    #             def B(B, b): pass
    #             def B(): pass
    #             def C(c): pass
    #             def C(D): pass
    #             def D(d, D): pass
    #             def D(E): pass
    #             def E(D): pass
    #             def E(B): pass
    #         # Now raised warnings get captured into `ws`.
    #         self.assertEqual(len(ws), 1)
    #         # print(ws)

        
if __name__ == '__main__':
    unittest.main()
    # pass
