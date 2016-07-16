import pprint as pp

from metaparse import *

ids = []

class LRVal(metaclass=cfg):

    # Lexical elements in attribute form:
    #
    #   <lex-name> = <re-pattern>
    #
    EQ   = r'='
    STAR = r'\*'
    ID   = r'[_a-zA-Z]\w*'

    # Rules in method form:
    #
    #   def <symbol> (<symbol> ... ):       # Syntatic rule by signature
    #       <do-sth> ...                    # Semantics in pure Python code!
    #       return <symbol-value>
    #
    def S(L, EQ, R):
        print('Got ids:', ids)
        print('assign %s to %s' % (R, L))
        ids.clear()
        
    def S(R):
        print('Got ids:', ids)
        return ('expr', R)

    def L(STAR, R):
        return ('REF', R)
    def L(ID):
        ids.append(ID)
        return ID

    def R(L):
        return L



class G_IfThenElse(metaclass=cfg):

    IF = r'if'
    THEN = r'then'
    ELSE = r'else'
    EXPR = r'\d+'
    SINGLE = r'\w+'

    def stmt(IF, EXPR, THEN, stmt):
        return ('i-t', stmt)
    def stmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return ('i-t-e', stmt_1, stmt_2)
    def stmt(SINGLE):
        return SINGLE

# p_ite = Earley(G_IfThenElse)
# p_ite = GLR(G_IfThenElse)
p_lalr_ite = LALR(G_IfThenElse)
inp = 'if 1 then if 2 then if 3 then a else b else c'
# # p_ear_ite.parse('if 1 then if 2 then if 3 then 4 else 5 else 6')
# pp.pprint([*p_ite.tokenize(inp, True)])

# res_many = p_ite.parse_many(inp)
# res_many = p_ite.interpret_many(inp) 
# pp.pprint(res_many)

res = p_lalr_ite.interpret(inp)
pp.pprint(res)
