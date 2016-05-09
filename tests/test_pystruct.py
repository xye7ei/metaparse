import preamble

from lalr import lalr
from earley import earley

# class PyStructReader(metaclass=earley):
class PyStructReader(metaclass=lalr):

    """
    Grammar for python object and built-in container types.
    """
    
    id = r'[A-Za-z_]\w*'
    comma = r','
    colon = r':'
    pl = r'\('
    pr = r'\)'
    bl = r'\['
    br = r'\]'
    Bl = r'\{'
    Br = r'\}'

    def Obj(id)                      : return ('Sym', id)
    def Obj(Lst)                     : return Lst
    def Obj(Tpl)                     : return Tpl
    def Obj(Dic)                     : return Dic
    def Obj(Set)                     : return Set

    def Tpl(pl, Terms, pr)           : return ('Tpl', Terms)
    def Lst(bl, Terms, br)           : return ('Lst', Terms)
    def Set(Bl, Terms, Br)           : return ('Set', Terms)
    def Dic(Bl, DTerms, Br)          : return ('Dic', DTerms)

    def Terms(Obj, comma, Terms)     : return [Obj] + Terms
    def Terms(Obj)                   : return [Obj]
    def Terms()                      : return []

    def DTerms(DTerm, comma, DTerms) : return [DTerm] + DTerms
    def DTerms(DTerm)                : return [DTerm]
    def DTerms()                     : return []

    def DTerm(Obj_1, colon, Obj_2)   : return (Obj_1, Obj_2)


if __name__ == '__main__':
    rd = PyStructReader
    import pprint as pp
    pp.pprint(rd.rules)
    # pp.pprint(rd.parse('[]'))
    # pp.pprint(rd.parse('[a, b,]'))
    # pp.pprint(rd.parse('[(a, b), c, {x : y, z : w}]'))
    # pp.pprint(rd.interprete('[(a, b), c, {x : y, z : w}]'))
    assert rd.interprete('a') == \
        ('Sym', 'a')
    assert rd.interprete('{(a, b), c, {e, f}}') == \
        ('Set', [('Tpl', [('Sym', 'a'), ('Sym', 'b')]),
                 ('Sym', 'c'),
                 ('Set', [('Sym', 'e'), ('Sym', 'f')])])
    # rd.parse('[(a, b), c, {x : y, z : w}]')
    # pp.pprint(rd.interprete('[]'))
    # pp.pprint(rd.interprete('[a, b,]'))
    # pp.pprint(rd.interprete('[(a, b), c, {x : y, z : w}]'))

