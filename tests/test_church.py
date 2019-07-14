from metaparse import LALR

class LangChurch(metaclass=LALR.meta):

    """
    Grammar for interpreting Church-Numerals.
    """

    ZERO = r'zero'
    SUCC = r'succ'

    def num(ZERO):
        return 0

    def num(SUCC, num):
        return num + 1


import unittest
class Test(unittest.TestCase):
    def test_church(self):
        self.assertEqual(LangChurch.interpret('zero')                ,  0)
        self.assertEqual(LangChurch.interpret('succ zero')           ,  1)
        self.assertEqual(LangChurch.interpret('succ succ zero')      ,  2)
        self.assertEqual(LangChurch.interpret('succ succ succ zero') ,  3)

if __name__ == '__main__':
    unittest.main()
