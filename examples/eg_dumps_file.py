## This file is generated. Do not modify.

## Lexer$BEGIN

lex2pats = \
    [('IGNORED', '\\s+'),
     ('EQ', '='),
     ('NUM', '[1-9]\\d*'),
     ('ID', '[_a-zA-Z]\\w*'),
     ('POW', '\\*\\*'),
     ('MUL', '\\*'),
     ('ADD', '\\+'),
     ('ERROR', '.')]

lex_handler_sources = \
    {'NUM': b'\xe3\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00'
            b'\x00C\x00\x00\x00s\n\x00\x00\x00t\x00\x00|\x00\x00\x83\x01\x00S'
            b')\x01N)\x01\xda\x05float)\x01\xda\x03lex\xa9\x00r\x03\x00'
            b'\x00\x00\xfa\x05<ast>\xda\x03NUM\x07\x00\x00\x00s\x02\x00\x00\x00\x00'
            b'\x01'}

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
    [b'\xe3\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00'
     b'\x00C\x00\x00\x00s\x04\x00\x00\x00|\x00\x00S)\x01N\xa9\x00)\x01\xda\x01x'
     b'r\x01\x00\x00\x00r\x01\x00\x00\x00\xfa8c:\\users\\shellay\\documents\\githu'
     b'b\\metaparse\\metaparse.py\xda\x07id_func\x91\x00\x00\x00s\x02\x00'
     b'\x00\x00\x00\x01',
     b'\xe3\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x03\x00\x00'
     b'\x00C\x00\x00\x00s\x0e\x00\x00\x00|\x02\x00t\x00\x00|\x00\x00<d\x00\x00S'
     b')\x01N)\x01\xda\x05table)\x03\xda\x02ID\xda\x02EQ\xda\x04expr\xa9\x00r\x05'
     b'\x00\x00\x00\xfa\x05<ast>\xda\x06assign\x0f\x00\x00\x00s\x02\x00\x00\x00\x00'
     b'\x01',
     b'\xe3\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00'
     b'\x00C\x00\x00\x00s\x04\x00\x00\x00|\x00\x00S)\x01N\xa9\x00)\x01\xda\x03N'
     b'UMr\x01\x00\x00\x00r\x01\x00\x00\x00\xfa\x05<ast>\xda\x04expr\x12\x00\x00'
     b'\x00s\x02\x00\x00\x00\x00\x01',
     b'\xe3\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00'
     b'\x00C\x00\x00\x00s\x08\x00\x00\x00t\x00\x00|\x00\x00\x19S)\x01N)\x01\xda'
     b'\x05table)\x01\xda\x02ID\xa9\x00r\x03\x00\x00\x00\xfa\x05<ast>\xda\x04expr'
     b'\x15\x00\x00\x00s\x02\x00\x00\x00\x00\x01',
     b'\xe3\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x02\x00\x00'
     b'\x00C\x00\x00\x00s\x08\x00\x00\x00|\x00\x00|\x02\x00\x17S)\x01N\xa9\x00)'
     b'\x03\xda\x06expr_1\xda\x03ADD\xda\x06expr_2r\x01\x00\x00\x00r'
     b'\x01\x00\x00\x00\xfa\x05<ast>\xda\x04expr\x18\x00\x00\x00s\x02\x00'
     b'\x00\x00\x00\x01',
     b'\xe3\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x02\x00\x00'
     b'\x00C\x00\x00\x00s\x08\x00\x00\x00|\x00\x00|\x02\x00\x14S)\x01N\xa9\x00)'
     b'\x03\xda\x04expr\xda\x03MUL\xda\x06expr_1r\x01\x00\x00\x00r\x01\x00'
     b'\x00\x00\xfa\x05<ast>r\x02\x00\x00\x00\x1b\x00\x00\x00s\x02\x00\x00\x00\x00'
     b'\x01',
     b'\xe3\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x02\x00\x00'
     b'\x00C\x00\x00\x00s\x08\x00\x00\x00|\x00\x00|\x02\x00\x13S)\x01N\xa9\x00)'
     b'\x03\xda\x04expr\xda\x03POW\xda\x06expr_1r\x01\x00\x00\x00r\x01\x00'
     b'\x00\x00\xfa\x05<ast>r\x02\x00\x00\x00\x1e\x00\x00\x00s\x02\x00\x00\x00\x00'
     b'\x01']

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
