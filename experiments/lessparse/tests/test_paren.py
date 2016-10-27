import preamble

from metaparse import *

class Gparen(metaclass=LALR.meta):

    """
    Grammar for matching arbitrary paired parenthesises.
    """

    END   = r'\$'
    LEFT  = r'\('
    RIGHT = r'\)'


    def top(pair):
        return pair


    def pair(LEFT, pair_1, RIGHT, pair_2):
        return '<' + pair_1 + '>' + pair_2


    def pair():
        return ''


from unittest import main, TestCase

class Test(TestCase):

    def test_paren(self):

        assert Gparen.interpret('()') == '<>'
        assert Gparen.interpret('( ( ) )') == '<<>>'
        assert Gparen.interpret('( ( ) ) ( )') == '<<>><>'

if __name__ == '__main__':
    # import pprint as pp
    # s = Gparen.parse('( ( ) ) ( )')
    # pp.pprint(s)
    main()
