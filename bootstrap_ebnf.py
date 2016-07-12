from metaparse import *

class EBNF(metaclass=WLL1):

    CMMT   = r'\(\*[^(\*\))]*\*\)'

    DIGIT  = r'[0-9]'
    L1     = r'\('
    R1     = r'\)'
    L2     = r'\['
    R2     = r'\]'
    L3     = r'\{'
    R3     = r'\}'
    TMSTR  = r'\"[^\"]*\"'
    SPEC   = r'\?[^\?]*\?'
    
    ID     = r''
