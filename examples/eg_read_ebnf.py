import preamble
from metaparse import *
from pprint import pprint


class EBNF(metaclass=LALR.meta):

    ID    = r'[a-zA-Z]\w+'
    TERM1 = r'\'[^\']*\''
    TERM2 = r'\"[^\"]*\"'

    DRV = r'='
    ALT = r'\|'
    CON = r','
    SEMI = r';'

    L  = r'\(' ; R  = r'\)'
    Lb = r'\[' ; Rb = r'\]'
    LB = r'\{' ; RB = r'\}'

    def grammar(rules):
        return rules

    def rules(rules, rule):
        rules.append(rule)
        return rules
    def rules():
        return []

    def rule(lhs, DRV, rhs, SEMI):
        return (Symbol(lhs), rhs)

    def lhs(ID):
        return ID
    def rhs(alts):
        return alts

    def alts(alts, ALT, seq):
        alts.append(seq)
        return alts
    def alts(seq):
        return [seq]

    def seq(seq, CON, expr):
        return seq + (expr,)
    def seq(expr):
        return (expr,)

    def expr(ID): return Symbol(ID)
    def expr(term): return term[1:-1]
    def expr(opt): return opt
    def expr(rep): return rep
    def expr(grp): return grp

    def term(TERM1): return TERM1
    def term(TERM2): return TERM2

    def grp(L, alts, R): return alts
    def opt(Lb, alts, Rb): return (Symbol('?'), alts)
    def rep(LB, alts, RB): return (Symbol('*'), alts)


inp = """
letter = "A" | "B" | "C" | "D" | "E" | "F" | "G"
       | "H" | "I" | "J" | "K" | "L" | "M" | "N"
       | "O" | "P" | "Q" | "R" | "S" | "T" | "U"
       | "V" | "W" | "X" | "Y" | "Z" | "a" | "b"
       | "c" | "d" | "e" | "f" | "g" | "h" | "i"
       | "j" | "k" | "l" | "m" | "n" | "o" | "p"
       | "q" | "r" | "s" | "t" | "u" | "v" | "w"
       | "x" | "y" | "z" ;
digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
symbol = "[" | "]" | "{" | "}" | "(" | ")" | "<" | ">"
       | "'" | '"' | "=" | "|" | "." | "," | ";" ;
character = letter | digit | symbol | "_" ;

identifier = letter , { letter | digit | "_" } ;
terminal = "'" , character , { character } , "'"
         | '"' , character , { character } , '"' ;

lhs = identifier ;
rhs = identifier
     | terminal
     | "[" , rhs , "]"
     | "{" , rhs , "}"
     | "(" , rhs , ")"
     | rhs , "|" , rhs
     | rhs , "," , rhs ;

rule = lhs , "=" , rhs , ";" ;
grammar = { rule } ;
"""

tr = EBNF.parse(inp)
e = EBNF.interpret(inp)

# pprint(tr)
pprint(e)
