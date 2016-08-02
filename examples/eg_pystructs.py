import preamble
import unittest

from metaparse import *
# from earley import earley

# class PyStructReader(metaclass=earley):
class PyStructReader(metaclass=LALR.meta):

    """
    Grammar for python object and built-in container types.
    """

    l1 = r'\('
    r1 = r',?\s*\)'
    l2 = r'\['
    r2 = r',?\s*\]'
    l3 = r'\{'
    r3 = r',?\s*\}'
    comma = r','
    colon = r':'
    id = r'[A-Za-z_]\w*'

    def Obj(id)                      : return ('Sym', id)

    def Obj(Lst)                     : return Lst
    def Obj(Tpl)                     : return Tpl
    def Obj(Dic)                     : return Dic
    def Obj(Set)                     : return Set

    def Tpl(l1, Objs, r1)            : return ('Tpl', Objs)
    def Lst(l2, Objs, r2)            : return ('Lst', Objs)
    def Set(l3, Obj, Objs, r3)       : return ('Set', [Obj] + Objs) # 'Set' contains at least one object
    def Dic(l3, DTerms, r3)          : return ('Dic', DTerms)

    def Objs(Objs, comma, Obj)       : return Objs + [Obj]
    def Objs(Obj)                    : return [Obj]
    def Objs()                       : return []

    def DTerms(DTerms, comma, DTerm) : return DTerms + [DTerm]
    def DTerms(DTerm)                : return [DTerm]
    def DTerms()                     : return []

    def DTerm(Obj_1, colon, Obj_2)   : return (Obj_1, Obj_2)


target = PyStructReader.interpret

class TestPyStructParser(unittest.TestCase):

    def test_empty_list(self):
        r = target('[]')
        self.assertEqual(r, ('Lst', []))

    def test_empty_dict(self):
        r = target('{}')
        self.assertEqual(r, ('Dic', []))

    def test_symbol(self):
        self.assertEqual(target('a'), ('Sym', 'a'))

    def test_normal_set(self):
        self.assertEqual(
            target('{(a, b), c, {e, f}}'),
            ('Set', [('Tpl', [('Sym', 'a'), ('Sym', 'b')]),
                     ('Sym', 'c'),
                     ('Set', [('Sym', 'e'), ('Sym', 'f')])]))

    def test_normal_dict(self):
        self.assertEqual(
            target('[{a: b}, {c}, {x: y, z: [a]}]'),
            ('Lst', [('Dic', [(('Sym', 'a'), ('Sym', 'b'))]),
                     ('Set', [('Sym', 'c')]),
                     ('Dic', [(('Sym', 'x'), ('Sym', 'y')),
                              (('Sym', 'z'), ('Lst', [('Sym', 'a')]))])]))


if __name__ == '__main__':
    unittest.main()
