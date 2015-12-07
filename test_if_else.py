from grammar import cfg
from earley import Earley
from lalr import LALR
from glr import GLR

import pprint as pp

class Gif(metaclass=cfg):

    """
    Grammar having 'dangling-else' ambiguity.
    This grammar does not lead LALR parser to raise any conflicts.
    Does absence of conflicts mean absence of ambiguity!?
    - NO, by trying the progagation process by hand it can be found
      the conflicts arise by state 6

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

    def ifstmt(IF, EXPR, THEN, stmt1, ELSE, stmt2):
        return ('if-then-else', EXPR, stmt1, stmt2) 

    def ifstmt(IF, EXPR, THEN, stmt):
        return ('if-then', EXPR, stmt)


if __name__ == '__main__':

    print('Waiting for LALR to report CONFLICTs...')
    plalr = LALR(Gif)
    pglr = GLR(Gif)
    # pear = Earley(Gif)

    import timeit

    # Errored input
    inp = 'if e then if e then if e then else s'
    print('\n\n\nGLR parsing "{}"'.format(inp))
    pglr.parse(inp)
    # timeit.timeit('pglr.parse(inp)')

    # Right input - 3 parse trees
    inp = 'if e then if e then if e then s else s'
    print('\n\n\nGLR parsing for 3 parse trees "{}"'.format(inp))
    pglr.parse(inp)
    # print(timeit.timeit('30**30**30'))

    # Right input - 4 parse trees
    inp = 'if e then if e then if e then if e then s else s'

