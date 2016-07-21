import preamble
import unittest

from metaparse import *

import pprint as pp

class Gif(metaclass=cfg):

    """
    Grammar having 'dangling-else' ambiguity.
    This grammar should lead LALR parser to raise conflicts.

Ks
(0, [(stmt^ -> .stmt)]),
(1, [(stmt^ -> stmt.)]),
(2, [(stmt -> ifstmt.)]),
(3, [(stmt -> SINGLE.)]),
(4, [(ifstmt -> IF.EXPR THEN stmt ELSE stmt), (ifstmt -> IF.EXPR THEN stmt)]),
(5, [(ifstmt -> IF EXPR.THEN stmt ELSE stmt), (ifstmt -> IF EXPR.THEN stmt)]),
(6, [(ifstmt -> IF EXPR THEN.stmt ELSE stmt), (ifstmt -> IF EXPR THEN.stmt)]),
(7, [(ifstmt -> IF EXPR THEN stmt.ELSE stmt), (ifstmt -> IF EXPR THEN stmt.)]),
(8, [(ifstmt -> IF EXPR THEN stmt ELSE.stmt)]),
(9, [(ifstmt -> IF EXPR THEN stmt ELSE stmt.)])

GOTO
(0, {'IF': 4, 'SINGLE': 3, 'ifstmt': 2, 'stmt': 1}),
(1, {}),
(2, {}),
(3, {}),
(4, {'EXPR': 5}),
(5, {'THEN': 6}),
(6, {'IF': 4, 'SINGLE': 3, 'ifstmt': 2, 'stmt': 7}),
(7, {'ELSE': 8}),
(8, {'IF': 4, 'SINGLE': 3, 'ifstmt': 2, 'stmt': 9}),
(9, {})

ACTION
(0, {'IF'     : ('shift', 4),
     'SINGLE' : ('shift', 3)}),
(1, {'END'    : ('accept', None)}),
(2, {'ELSE'   : ('reduce', (stmt -> ifstmt.)),
     'END'    : ('reduce', (stmt -> ifstmt.))}),
(3, {'ELSE'   : ('reduce', (stmt -> SINGLE.)),
     'END'    : ('reduce', (stmt -> SINGLE.))}),
(4, {'EXPR'   : ('shift', 5)}),
(5, {'THEN'   : ('shift', 6)}),
(6, {'IF'     : ('shift', 4),
     'SINGLE' : ('shift', 3)}),
(7, {'ELSE'   : ('shift', 8),
     'END'    : ('reduce', (ifstmt -> IF EXPR THEN stmt.))}),
(8, {'IF'     : ('shift', 4),
     'SINGLE' : ('shift', 3)}),
(9, {'ELSE'   : ('reduce', (ifstmt -> IF EXPR THEN stmt ELSE stmt.)),
     'END'    : ('reduce', (ifstmt -> IF EXPR THEN stmt ELSE stmt.))}) 

! LALR-Conflict raised:
  - in ACTION[7]: 
    {'ELSE': ('shift', 8), 'END': ('reduce', (ifstmt -> IF EXPR THEN stmt.))}
  - conflicting action on token 'ELSE': 
    {'ELSE': ('reduce', (ifstmt -> IF EXPR THEN stmt.))}

    """

    IF     = r'if'
    THEN   = r'then'
    ELSE   = r'else'
    EXPR   = r'e'
    SINGLE = r's'


    def stmt(ifstmt):
        return ifstmt 

    def stmt(SINGLE):
        return SINGLE 

    def ifstmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return ('ite', EXPR, stmt_1, stmt_2) 

    def ifstmt(IF, EXPR, THEN, stmt):
        return ('it', EXPR, stmt)


class TestLRGrammar(unittest.TestCase):

    def test_LALR_report(self):
        """LALR parser should report conflicts for ambiguous grammar! """
        with self.assertRaises(ParserError):
            LALR(Gif)

    def test_many(self):

        p_ear = Earley(Gif)
        p_gll = GLL(Gif)
        p_glr = GLR(Gif)

        inp = 'if e then if e then if e then s else s else s'

        p1 = sorted(p_ear.interpret_many(inp))
        p2 = sorted(p_glr.interpret_many(inp))
        p3 = sorted(p_gll.interpret_many(inp))

        pp.pprint(p1)
        pp.pprint(p2)
        pp.pprint(p3)

        self.assertEqual(p1, p2)
        self.assertEqual(p2, p3)
        self.assertEqual(len(p1), 3)


if __name__ == '__main__':

    # unittest.main()

    LALR(Gif)
