import preamble
from metaparse import *

@meta
class WLL1(ParserDeterm):
    """Weak-LL(1)-Parser.

    Since 'strong'-LL(1) grammar parser includes the usage of FOLLOW
    set, which is only heuristically helpful for the recognitive
    capability when handling NULLABLE rules, this parser suppress the
    need of FOLLOW.

    When deducing a NULLABLE nonterminal A with some lookahead a, if a
    does not belong to any FIRST of A's alternatives, then the NULL
    alternative is chosen. In other words, all terminals not in
    FIRST(A) leads to the prediction (as well as immediate reduction)
    of (A -> ε) in the predictive table.

    This variation allows predicting (A -> ε) even when lookahead a is
    not in FOLLOW, which means this parser will postpone the
    recognition error compared to strong-LL(1) parser.

    """

    def __init__(self, grammar):
        self.grammar = grammar
        self.lexer = Lexer.from_grammar(grammar)
        self.semans = grammar.semans
        self._calc_ll1_table()

    def _calc_ll1_table(self):
        G = self.grammar
        table = self.table = {}
        for r, rule in enumerate(G.rules):
            lhs, rhs = rule
            if lhs not in table:
                table[lhs] = {}
            # NON-NULL rule
            if rhs:
                for a in G.first_of_seq(rhs, EPSILON):
                    if a is EPSILON:
                        pass
                    elif a in table[lhs]:
                        raise GrammarError('Not simple LL(1) grammar! ')
                    else:
                        table[lhs][a] = rule
            # NULL rule
            # This rule tends to be tried when
            # the lookahead doesn't appear in
            # other sibling rules.
            else:
                pass

    def parse(self, inp, interp=False):
        """The process is exactly the `translate' process of a ParseTree.

        """
        # Backtracking is yet supported
        # Each choice should be deterministic
        push = list.append
        pop = list.pop
        G = self.grammar
        pstack = self.pstack = []
        table = self.table
        toker = enumerate(self.lexer.tokenize(inp, with_end=True))
        pstack.append(G.rules[0].lhs)
        argstack = []
        try:
            k, tok = next(toker)
            while pstack:
                actor = pop(pstack)
                at, look, tokval = tok
                # Reduction
                if isinstance(actor, Rule):
                    args = []
                    # Pop the size of args, conclude subtree
                    # for prediction made before
                    for _ in actor.rhs:
                        args.insert(0, pop(argstack))
                    if interp:
                        arg1 = actor.seman(*args)
                    else:
                        arg1 = ParseTree(actor, args)
                    # Finish - no prediction in stack
                    # Should declare end-of-input
                    if not pstack:
                        return arg1
                    else:
                        push(argstack, arg1)
                # Make prediction on nonterminal
                elif actor in G.nonterminals:
                    if look in table[actor]:
                        pred = table[actor][look]
                        # Singal for reduction
                        push(pstack, pred)
                        # Push symbols into prediction-stack,
                        # last symbol first in.
                        for x in reversed(pred.rhs):
                            push(pstack, x)
                    # !!! Heuristically do epsilon-reduction when no
                    # viable lookahead found
                    elif actor in G.NULLABLE:
                        for r0 in G.rules:
                            if r0.lhs == actor and not r0.rhs:
                                if interp:
                                    argstack.append(r0.seman())
                                else:
                                    argstack.append(ParseTree(r0, []))
                    # Recognition failed, ignore
                    else:
                        raise ParserError('No production found.')
                # Try match terminal
                else:
                    if actor == look:
                        if interp:
                            argstack.append(tokval)
                        else:
                            argstack.append(tok)
                    k, tok = next(toker)

        except StopIteration:
            raise ParserError('No enough tokens to complete parsing.')

