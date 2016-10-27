lex2pats = \
    [('IGNORED', '%'),
     ('COMMA', ','),
     ('LEFT', '\\('),
     ('RIGHT', '\\)'),
     ('IGNORED', '\\s+'),
     ('SYMBOL', '[_a-zA-Z]\\w*'),
     ('UNKNOWN', '&'),
     ('NUMBER', '[1-9]\\d*(\\.\\d*)?')]

handlers = \
    [None,
     None,
     None,
     None,
     None,
     None,
     None,
     b'\xe3\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00'
     b'\x00C\x00\x00\x00s\n\x00\x00\x00t\x00\x00|\x00\x00\x83\x01\x00S)\x01N)'
     b'\x01\xda\x03int)\x01\xda\x03val\xa9\x00r\x03\x00\x00\x00\xfaOc:/Users/Shell'
     b'ay/Documents/GitHub/metaparse/experiments/lessparse/test_basic.py\xda\x01_'
     b'g\x00\x00\x00s\x02\x00\x00\x00\x00\x02']

rules = \
    [('sexp^', ('sexp',)),
     ('sexp', ('atom',)),
     ('sexp', ('LEFT', 'slist', 'RIGHT')),
     ('slist', ()),
     ('slist', ('slist', 'sexp')),
     ('atom', ('NUMBER',)),
     ('atom', ('SYMBOL',))]

ACTION1 = \
    [{'LEFT': ('shift', 3), 'NUMBER': ('shift', 4), 'SYMBOL': ('shift', 5)},
     {'\x03': ('reduce', 0)},
     {'\x03': ('reduce', 1),
      'LEFT': ('reduce', 1),
      'NUMBER': ('reduce', 1),
      'RIGHT': ('reduce', 1),
      'SYMBOL': ('reduce', 1)},
     {'LEFT': ('reduce', 3),
      'NUMBER': ('reduce', 3),
      'RIGHT': ('reduce', 3),
      'SYMBOL': ('reduce', 3)},
     {'\x03': ('reduce', 5),
      'LEFT': ('reduce', 5),
      'NUMBER': ('reduce', 5),
      'RIGHT': ('reduce', 5),
      'SYMBOL': ('reduce', 5)},
     {'\x03': ('reduce', 6),
      'LEFT': ('reduce', 6),
      'NUMBER': ('reduce', 6),
      'RIGHT': ('reduce', 6),
      'SYMBOL': ('reduce', 6)},
     {'LEFT': ('shift', 3),
      'NUMBER': ('shift', 4),
      'RIGHT': ('shift', 7),
      'SYMBOL': ('shift', 5)},
     {'\x03': ('reduce', 2),
      'LEFT': ('reduce', 2),
      'NUMBER': ('reduce', 2),
      'RIGHT': ('reduce', 2),
      'SYMBOL': ('reduce', 2)},
     {'LEFT': ('reduce', 4),
      'NUMBER': ('reduce', 4),
      'RIGHT': ('reduce', 4),
      'SYMBOL': ('reduce', 4)}]

GOTO = \
    [{'LEFT': 3, 'NUMBER': 4, 'SYMBOL': 5, 'atom': 2, 'sexp': 1},
     {},
     {},
     {'slist': 6},
     {},
     {},
     {'LEFT': 3, 'NUMBER': 4, 'RIGHT': 7, 'SYMBOL': 5, 'atom': 2, 'sexp': 8},
     {},
     {}]

semans = \
    [b'\xe3\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00'
     b'\x00C\x00\x00\x00s\x04\x00\x00\x00|\x00\x00S)\x01N\xa9\x00)\x01\xda\x01x'
     b'r\x01\x00\x00\x00r\x01\x00\x00\x00\xfaNc:\\Users\\Shellay\\Documents\\GitHu'
     b'b\\metaparse\\experiments\\lessparse\\metaparse.py\xda\x08identity'
     b'*\x00\x00\x00s\x02\x00\x00\x00\x00\x01',
     b'\xe3\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00'
     b'\x00C\x00\x00\x00s\x04\x00\x00\x00|\x00\x00S)\x01N\xa9\x00)\x01\xda\x04atomr'
     b'\x01\x00\x00\x00r\x01\x00\x00\x00\xfaOc:/Users/Shellay/Documents/GitHub/met'
     b'aparse/experiments/lessparse/test_basic.py\xda\x04sexpk\x00\x00\x00'
     b's\x02\x00\x00\x00\x00\x02',
     b'\xe3\x03\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x01\x00\x00'
     b'\x00C\x00\x00\x00s\x04\x00\x00\x00|\x01\x00S)\x01N\xa9\x00)\x03\xda\x04L'
     b'EFT\xda\x05slist\xda\x05RIGHTr\x01\x00\x00\x00r\x01\x00\x00\x00\xfaOc:/User'
     b's/Shellay/Documents/GitHub/metaparse/experiments/lessparse/test_basic.py'
     b'\xda\x04sexpn\x00\x00\x00s\x02\x00\x00\x00\x00\x02',
     b'\xe3\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00'
     b'\x00C\x00\x00\x00s\x04\x00\x00\x00g\x00\x00S)\x01N\xa9\x00r\x01\x00\x00\x00'
     b'r\x01\x00\x00\x00r\x01\x00\x00\x00\xfaOc:/Users/Shellay/Documents/GitHub/me'
     b'taparse/experiments/lessparse/test_basic.py\xda\x05slists\x00\x00\x00s\x02'
     b'\x00\x00\x00\x00\x02',
     b'\xe3\x02\x00\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x02\x00\x00'
     b'\x00C\x00\x00\x00s\x11\x00\x00\x00|\x00\x00j\x00\x00|\x01\x00\x83'
     b'\x01\x00\x01|\x00\x00S)\x01N)\x01\xda\x06append)\x02\xda\x05slist\xda\x04s'
     b'exp\xa9\x00r\x04\x00\x00\x00\xfaOc:/Users/Shellay/Documents/GitHub/metapa'
     b'rse/experiments/lessparse/test_basic.pyr\x02\x00\x00\x00v\x00\x00\x00'
     b's\x04\x00\x00\x00\x00\x02\r\x01',
     b'\xe3\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00'
     b'\x00C\x00\x00\x00s\x04\x00\x00\x00|\x00\x00S)\x01N\xa9\x00)\x01\xda\x06NUMBE'
     b'Rr\x01\x00\x00\x00r\x01\x00\x00\x00\xfaOc:/Users/Shellay/Documents/GitHub/m'
     b'etaparse/experiments/lessparse/test_basic.py\xda\x04atom{\x00\x00\x00s\x02'
     b'\x00\x00\x00\x00\x02',
     b'\xe3\x01\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x01\x00\x00'
     b'\x00C\x00\x00\x00s\x04\x00\x00\x00|\x00\x00S)\x01N\xa9\x00)\x01\xda\x06SYMBO'
     b'Lr\x01\x00\x00\x00r\x01\x00\x00\x00\xfaOc:/Users/Shellay/Documents/GitHub/m'
     b'etaparse/experiments/lessparse/test_basic.py\xda\x04atom~\x00\x00\x00s\x02'
     b'\x00\x00\x00\x00\x02']
