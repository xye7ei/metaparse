# -*- coding: utf-8 -*-
import preamble

from metaparse import grammar, LALR

@grammar
def S1():

    """Given grammar:

    S = A | B | C
    A = a
    B = a + a
    C = A D
    D = + a

    The R/S conflict in state [(A = a.), (B = a.+ a)] cannot be
    resolved with any precedence for PLUS.

    """

    a = r'a'
    PLUS = r'\+'

    def S(A): pass
    def S(B): pass
    def S(C): pass

    def A(a): pass

    def B(a, PLUS, a_1): pass
    def C(A, D): pass
    def D(PLUS, A): pass


@grammar
def S2():

    """Given grammar:

    S = A | B
    A = a
    B = a

    The R/R conflicts in state [(A = a.), (B = a.)] cannot be
    resolved.

    """
    a = r'a'

    def S(A): pass
    def S(B): pass
    def A(a): pass
    def B(a): pass


@grammar
def S3():

    """Given grammar:

    E = E + E
    E = E * E
    E = NUM
    
    The S/R conflict in state [(E^ = E.), (E = E.+ E), (E = E.* E)] need
    to be resolved by specifying precedence.

    """
    NUM = r'[1-9][0-9]*'
    def NUM(val):
        return int(val)

    PLUS = r'\+'
    TIMES = r'\*'

    def E(E, PLUS, E_1):
        return E + E_1
    def E(E, TIMES, E_1):
        return E * E_1
    def E(NUM):
        return NUM

p1 = LALR(S1)
# p2 = LALR(S2)
# p3 = LALR(S3)
# print(p3.interpret('3 + 2 * 5'))
