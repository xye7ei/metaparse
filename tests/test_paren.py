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


    def pair(LEFT, pair1, RIGHT, pair2):
        return '<' + pair1 + '>' + pair2


    def pair():
        return ''

s = Gparen.parse('( ( ) ) ( )')
v = Gparen.eval('( ( ) ) ( )')
import pprint as pp
pp.pprint(s)
pp.pprint(v)
