import preamble
import unittest
from metaparse import *


class G(metaclass=cfg):
    'A grammar which is LR(1) but not LALR(1).'

    a = r'a'
    b = r'b'
    c = r'c'
    d = r'd'
    e = r'e'

    def S(a, A, d): return
    def S(b, B, d): return
    def S(a, B, e): return
    def S(b, A, e): return

    def A(c): return c
    def B(c): return c


class TestConflicts(unittest.TestCase):

    def test_conflicts(self):
        self.assertRaises(ParserError, LALR, G)


if __name__ == '__main__':
    unittest.main()
