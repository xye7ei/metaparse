## This file is generated. Do not modify.

## Lexer$BEGIN

lex2pats = \
    [('EQ', '='),
     ('NUM', '[1-9]\\d*'),
     ('ID', '[_a-zA-Z]\\w*'),
     ('POW', '\\*\\*'),
     ('MUL', '\\*'),
     ('ADD', '\\+'),
     ('IGNORED', '[ \\t\\n]'),
     ('ERROR', '.')]
lex_handler_sources = \
    {'NUM': "def NUM(lex: r'[1-9]\\d*'):\n    return float(lex)\n"}

## Lexer$END


## Parser$BEGIN

rules = \
    [('assign^', ('assign',)),
     ('assign', ('ID', 'EQ', 'expr')),
     ('expr', ('NUM',)),
     ('expr', ('ID',)),
     ('expr', ('expr', 'ADD', 'expr')),
     ('expr', ('expr', 'MUL', 'expr')),
     ('expr', ('expr', 'POW', 'expr'))]

seman_sources = \
    ['def id_func(x):\n    return x\n',
     'def assign(ID, EQ, expr):\n    table[ID] = expr\n',
     'def expr(NUM):\n    return NUM\n',
     'def expr(ID):\n    return table[ID]\n',
     'def expr(expr_1, ADD, expr_2):\n    return expr_1 + expr_2\n',
     'def expr(expr, MUL, expr_1):\n    return expr * expr_1\n',
     'def expr(expr, POW, expr_1):\n    return expr ** expr_1\n']

Ks = \
    [[(0, 0)],
     [(0, 1)],
     [(1, 1)],
     [(1, 2)],
     [(1, 3), (4, 1), (5, 1), (6, 1)],
     [(2, 1)],
     [(3, 1)],
     [(4, 2)],
     [(5, 2)],
     [(6, 2)],
     [(4, 1), (4, 3), (5, 1), (6, 1)],
     [(4, 1), (5, 1), (5, 3), (6, 1)],
     [(4, 1), (5, 1), (6, 1), (6, 3)]]

ACTION = \
    [{'ID': ('SHIFT', 2)},
     {'END': ('ACCEPT', 0)},
     {'EQ': ('SHIFT', 3)},
     {'ID': ('SHIFT', 6), 'NUM': ('SHIFT', 5)},
     {'ADD': ('SHIFT', 7),
      'END': ('REDUCE', 1),
      'MUL': ('SHIFT', 8),
      'POW': ('SHIFT', 9)},
     {'ADD': ('REDUCE', 2),
      'END': ('REDUCE', 2),
      'MUL': ('REDUCE', 2),
      'POW': ('REDUCE', 2)},
     {'ADD': ('REDUCE', 3),
      'END': ('REDUCE', 3),
      'MUL': ('REDUCE', 3),
      'POW': ('REDUCE', 3)},
     {'ID': ('SHIFT', 6), 'NUM': ('SHIFT', 5)},
     {'ID': ('SHIFT', 6), 'NUM': ('SHIFT', 5)},
     {'ID': ('SHIFT', 6), 'NUM': ('SHIFT', 5)},
     {'ADD': ('REDUCE', 4),
      'END': ('REDUCE', 4),
      'MUL': ('SHIFT', 8),
      'POW': ('SHIFT', 9)},
     {'ADD': ('REDUCE', 5),
      'END': ('REDUCE', 5),
      'MUL': ('REDUCE', 5),
      'POW': ('SHIFT', 9)},
     {'ADD': ('REDUCE', 6),
      'END': ('REDUCE', 6),
      'MUL': ('REDUCE', 6),
      'POW': ('REDUCE', 6)}]

GOTO = \
    [{'ID': 2, 'assign': 1},
     {},
     {'EQ': 3},
     {'ID': 6, 'NUM': 5, 'expr': 4},
     {'ADD': 7, 'MUL': 8, 'POW': 9},
     {},
     {},
     {'ID': 6, 'NUM': 5, 'expr': 10},
     {'ID': 6, 'NUM': 5, 'expr': 11},
     {'ID': 6, 'NUM': 5, 'expr': 12},
     {'ADD': 7, 'MUL': 8, 'POW': 9},
     {'ADD': 7, 'MUL': 8, 'POW': 9},
     {'ADD': 7, 'MUL': 8, 'POW': 9}]

## Parser$END
