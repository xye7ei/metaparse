import preamble

from earley import earley

class Gparen(metaclass=earley):

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


if __name__ == '__main__':
    # import pprint as pp
    # s = Gparen.parse('( ( ) ) ( )')
    # pp.pprint(s)
    assert Gparen.interpret('()') == '<>'
    assert Gparen.interpret('( ( ) )') == '<<>>'
    assert Gparen.interpret('( ( ) ) ( )') == '<<>><>'
