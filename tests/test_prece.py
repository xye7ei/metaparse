# -*- coding: utf-8 -*-
import preamble

import pprint as pp
from metaparse import LALR, cfg, GLR

# Global stuff
table = {}

class G_Calc(metaclass=cfg):

    # ===== Lexical patterns / Terminals =====
    # - A pattern is defined by Python regex literal.
    # - Patterns will be matched in given order when tokenizing.

    IGNORED = r' '              # Special token ignored by tokenizer.
    IGNORED = r'\t'             # Can add alternative patterns.

    POW = r'\*\*', 3            # Precedence of token (for LALR)
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    EQ  = r'='                  # Precedence is 0 by default.

    NUM = r'[1-9]\d*'
    def NUM(lex):                 # Handler for translating token value.
        return int(lex)

    ID  = r'[_a-zA-Z]\w*'       # Unhandled token yields literal value.

    # === Optional error handling for tokenizer ===
    # - If handler defined, token ERROR is ignored when tokenizing.
    # - Otherwise token ERROR is yielded.
    ERROR = r'#'
    def ERROR(lex):
        print("Error literal '{}'".format(lex))

    # ===== Syntactic/Semantic rules in SDT-style =====

    def assign(ID, EQ, expr):        # May rely on side-effect...
        table[ID] = expr

    def expr(NUM):                   # or return local results for purity
        return NUM

    def expr(ID):
        return table[ID]

    def expr(expr_1, ADD, expr_2):   # With TeX-subscripts, meaning (expr → expr₁ + expr₂)
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):     # Can ignore one of the subscripts.
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1


from metaparse import cfg


pCalc = LALR(G_Calc)

from unittest import main, TestCase


class Test(TestCase):

    def test_interp(self):
        # parse and tree
        t = pCalc.parse("x = 1 + 4 * 3 ** 2 + 5")
        # interpretation of tree
        t.translate()
        assert table == {'x': 42}

        # direct interpretation
        # pCalc.interpret("x = 1 + 4 * 3 ** 2 + 5")
        pCalc.interpret("y = 5 + x * 2")
        assert table == {'x': 42, 'y': 5 + 42 * 2}
        pCalc.interpret("z = 99")
        assert table == {'x': 42, 'y': 5 + 42 * 2, 'z': 99}


if __name__ == '__main__':
    main()
