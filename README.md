# parket

A python parser packet for studying and analyzing **Context Free
Grammars**(CFG), including Earley, GLR(0), LALR(1) parsers. Also LL(1) parser
based upon Parsing Expression Grammar(PEG) is being coorperated thereto.

Features:

## Object-Oriented grammar definition

Since the traditional way of using yacc/bison tool chains needs the definition
of grammar to be written as a separate files, which can be treated as a
**DSL**(Domain Specified Language) based upon C language, is too much an over-
complexed work for parsing simple CFGs in many practical cases handling
textual information for their structures.

### Metaprogramming support lexical/syntactic/semantic definition

Making use of high-level features of Python, writing a grammar can be almost
with the same the syntax as defining a class with methods. Syntactic rules
and corresponded semantic operation can be written at the same time.

Example of
```python
from grammar import cfg

class GArith(metaclass=cfg):

    # E -> E + T
    # E -> T
    # T -> T * F
    # T -> F
    # F -> NUMB
    # F -> ( E )

    PLUS  = r'\+'
    TIMES = r'\*'
    NUMB  = r'\d+'
    LEFT  = r'\('
    RIGHT = r'\)'


    def expr(expr, PLUS, term):
        return expr + term

    def expr(term):
        return term

    def term(term, TIMES, factor):
        return term * factor

    def term(factor):
        return factor

    def factor(atom):
        return atom

    def factor(LEFT, expr, RIGHT):
        return expr

    def atom(NUMB):
        return int(NUMB)

```

The key trick here is utilizing Python's reflective functionalities to
transform signatures of methods into pre-defined Rule objects.


## Context Free Grammar (CFG) Parsers

Based upon the common infrastucture of grammar definitions, it's easy to
extend various parser families to analyze characteristics of some grammar.

The result of parsing is by default a parse tree or even a parse forest,
with respect to deterministic/non-deterministic parser.

### Non-deterministic parsers.

Non-deterministic parsers are generally more powerful(they handle ambiguity
and conflicts in a canonical way preserving each feasible parsing state).
They return a parse forest by default.

Example of using Earley parser. A parse tree is represented as a tuple, while
a parse forest as a list of tuples.

```python

from earley import Earley

p = Earley(GArith)

inp = '3 + 2 * 5 + 11'

s = p.parse(inp)

```

Resulted parse tree as a singleton parse forest, since
Earley parser is non-deterministic:

```python

[('expr^',
  [('expr',
    [('expr',
      [('expr', [('term', [('factor', [('atom', ['3'])])])]),
       '+',
       ('term',
        [('term', [('factor', [('atom', ['2'])])]),
         '*',
         ('factor', [('atom', ['5'])])])]),
     '+',
     ('term', [('factor', [('atom', ['11'])])])])])]

```

Another wide used non-deterministic parser is GLR parser.

### Deterministic parsers.

Typically, a well-formed deterministic parser always produces either one parse
tree or a failure. It can handle only a subset of CFG according to their
specified characteristics.

But these grammars are always more efficient when parsing and can be applied
without generating a parse tree explicitly (commonly generating IR code during
the parsing process). Most widely used deterministic parser is the LALR(1)
parser, which is the kernel of yacc/bison program. The implementation of
LALR(1) is tricky and may lose its popularity due to its constraints and against
the advancing more powerful parser like LL(*).

A well-designed LALR(1) parser can report conflicts clearly. It is hand coded
in this packet for completion, thus to make comparisons between it and the GLR
parser.


## Expression based parsers

The Parsing Expression Grammar(PEG) is also winning its popularity for its
favor of writing grammars in EBNF way and its capability of handing grammars
out of CFG scope. The effective tools are like Haskell's *parsec* library,
based upon **Parser Combinators**.

Since EBNF may not be well conforming the expressive power of Python's
metaprogramming functionalities like above, it is not the topic covered in
this packet.

