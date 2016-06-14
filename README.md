# parket

This package provides the most simple and elegant way of getting parsing work
done. It aims to be the replacement of traditional *flex/yacc*-style toolset
in Python environment.

## Background

The initiative for creating this package is to support studying and analyzing
**Context Free Grammars(CFG)** in a easy way. Parsers like Earley, GLR(0),
LALR(1) parsers. Also LL(1) and *parsec* parser based upon Parsing Expression
Grammar(PEG) is being integrated but still not completed.


# Rationale

This package is remarkable for its amazing simplicity and elegancy thanks to
the extreme reflexibility of Python 3 language.

## About *flex/bison*

Traditional parsing work is mostly supported with the toolset
lex/yacc(flex/bison). Such toolset is found to be hard and complex to handle
even for experienced programmers who just want do some handy parsing. 

<!-- 
The biggest reason for that is the complexity raised by integrating the
*Backus-Naur-Form*-grammar as **Domain-Specific Language** into *C*-style
coding environment. The compilation process and maintainance of intermediate
files(**\*.l**, **\*.y**) are old-fashioned.
 -->

## About *parsec*

Fans of functional languages like *Haskell* often appreciate the powerful
**parsec** library for its expression-based translation mechanism. Pitifully,
the intrinsic LL(1)-nature and explicit usage of try-function comprise
significant limitations.


# OO-style grammar definition

Coincidently, there is some similarity between BNF-style grammar rule
definition and Python function signature. Moreover, the semantic
behavior corresponding to a rule can be represented by the function
body.

Given a examplar rule in *BISON*:
```c
exp:      NUM
        | exp exp '+'     { $$ = $1 + $2;    }
        | exp exp '-'     { $$ = $1 - $2;    }
        ...
        ;
```

the sematically equivalent representation in this package would be
```python

    PLUS = r'\+'
    MINUS = r'-'

    def exp(exp_1, exp_2, PLUS):
        return exp_1 + exp_2
    
    def exp(exp_1, exp_2, MINUS):
        return exp_1 - exp_2
```

## An example overview

While the lexical declaration can be written as _class-attribute_ declarations
and the rules can be written as _method_ declarations, a grammar instance for
interperting arithmetic expressions in this package can be written as follows:

```python
from grammar import cfg

class GArith(metaclass=cfg):

    # E ::= E + T
    # E ::= T
    # T ::= T * F
    # T ::= F
    # F ::= NUMB
    # F ::= ( E )

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

The key trick supporting this is `Python 3`'s reflective functionalities to
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

Another wide used non-deterministic parser is the GLR parser. But here
in this packet GLR are only implemented as GLR(0), which is only
capable for parsing LR(0) language. 

### Deterministic parsers.

Typically, a well-formed deterministic parser always produces either
one parse tree or a failure. It can handle only a subset of CFG
according to their specified characteristics.

But these grammars are always more efficient when parsing and can be
applied without generating a parse tree explicitly (commonly
generating IR code during the parsing process). Most widely used
deterministic parser is the LALR(1) parser, which is the kernel of
yacc/bison program. The implementation of LALR(1) is tricky and may
lose its popularity due to its constraints and against the advancing
more powerful parser like LL(*).

A well-designed LALR(1) parser can report conflicts clearly. 


## Expression based parsers

The Parsing Expression Grammar(PEG) is also winning its popularity for its
favor of writing grammars in EBNF way and its capability of handing grammars
out of CFG scope. The effective tools are like Haskell's *parsec* library,
based upon **Parser Combinators**.

Since EBNF may not be well conforming the expressive power of Python's
metaprogramming functionalities like above, it is not the topic covered in
this packet.

