from sys import path; path.append('..')

from grammar import cfg
from glr import GLR
from lalr import LALR, lalr
from earley import Earley

# Two ways to specify a parser for given Grammar:
# 
# - class G1(metaclass=cfg):
#   ...
#   lalrpar = LALR(*G1)
#   ...
#   - This serves for comparison of different Parsers,
#     since the grammar are shared by them and written
#     only once. 
# 
# - class lalrpar(metaclass=lalr):
#   ...
#   - This serves for single use of one specified Parser. 

class G(metaclass=cfg):

    a = r'a'
    b = r'b'
    c = r'c'
    d = r'd'
    e = r'e'

    def S(a, A, d): return
    def S(b, B, d): return
    def S(a, B, e): return
    def S(b, A, e): return

    def A(c): return c
    def B(c): return c

def test_comp():
    Gglr = GLR(*G)
    # Glalr = LALR(*G)
    Gear = Earley(*G)

    # list(enumerate(Gglr.Ks))
    # list(enumerate(Gglr.table))
    # list(enumerate(Gglr.ACTION))
    # list(enumerate(Gglr.GOTO))
    # list(zip(Gglr.Ks, Gglr.GOTO))

    rg = Gglr.parse('a c e') 
    re = Gear.parse('a c e')

def use_one():

    class G1(metaclass=lalr):

        a = r'a'
        b = r'b'
        c = r'c'
        d = r'd'
        e = r'e'

        def S(a, A, d): return
        def S(b, B, d): return
        def S(a, B, e): return
        def S(b, A, e): return

        def A(c): return c
        def B(c): return c
