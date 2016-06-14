# parket

This package provides the most simple and elegant way of getting parsing work
done. It aims to be the replacement of traditional *flex/yacc*-style toolset
in Python environment.

### Background

The initiative for creating this package is to support studying and analyzing
**Context Free Grammars(CFG)** in a easy way. Parsers like Earley, GLR(0),
LALR(1) parsers. Also LL(1) and *parsec* parser based upon Parsing Expression
Grammar(PEG) is being integrated but still not completed.


## Rationale

This package is remarkable for its amazing simplicity and elegancy thanks to
the extreme reflexibility of Python 3 language.

### About *flex/bison*

Traditional parsing work is mostly supported with the toolset
lex/yacc(flex/bison). Such toolset is found to be hard and complex to handle
even for experienced programmers who just want do some handy parsing. 

<!-- 
The biggest reason for that is the complexity raised by integrating the
*Backus-Naur-Form*-grammar as **Domain-Specific Language** into *C*-style
coding environment. The compilation process and maintainance of intermediate
files(**\*.l**, **\*.y**) are old-fashioned.
 -->

### About *parsec*

Fans of functional languages like *Haskell* often appreciate the powerful
**parsec** library for its expression-based translation mechanism. Pitifully,
the intrinsic LL(1)-nature and explicit usage of try-function comprise
significant limitations.


## Parser front-end: OO-style grammar definition

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
NUM = r'\d+'
PLUS = r'\+'
MINUS = r'-'

def exp(exp_1, exp_2, PLUS):
    return exp_1 + exp_2
    
def exp(exp_1, exp_2, MINUS):
    return exp_1 - exp_2
    
...
```

## An example overview

While the lexical declaration can be written as _class-attribute_ declarations
and the rules can be written as _method_ declarations, a grammar instance for
interperting arithmetic expressions in this package can be written as follows:

```python
from grammar import cfg
from lalr import LALR

class GArith(metaclass=cfg):

    IGNORED = r'\s'

    plus   = r'\+'
    times  = r'\*'
    number = r'\d+'
    left   = r'\('
    right  = r'\)'


    def Expr(Expr, plus, Term):
        return Expr + Term
    def Expr(Term):
        return Term

    def Term(Term, times, Factor):
        return Term * Factor
    def Term(Factor):
        return Factor

    def Factor(Atom):
        return Atom
    def Factor(left, Expr, right):
        return Expr

    def Atom(number):
        return int(number)

p = LALR(GArith)

#- produces parse tree 
p.parse('3 + 2 * (5 + 11)')
#- output: 
'''
('Expr',
 [('Expr', [('Term', [('Factor', [('Atom', ['3'])])])]),
  '+',
  ('Term',
   [('Term', [('Factor', [('Atom', ['2'])])]),
    '*',
    ('Factor',
     ['(',
      ('Expr',
       [('Expr', [('Term', [('Factor', [('Atom', ['5'])])])]),
        '+',
        ('Term', [('Factor', [('Atom', ['11'])])])]),
      ')'])])])
'''

#- direct interpretation
p.interprete('3 + 2 * (5 + 11)')
#- output: 35
```

The key trick supporting this style is `Python 3`'s reflection and 
metaprogramming functionalities. By using the module `inspect`, 
signatures of a method can be transformed into  a pre-defined grammar rule 
object and the method body can be referenced as  semantic object associated 
with this object.

 
## Parser back-end: Context Free Grammar (CFG) Parsers

Based upon the concepts above, grammar representations are easily translated
into grammar objects. Upon that, it is natural to prepare various parser
algorithms as the back-end for various purposes. Some of them perform parsing 
process directly (like **Earley's Parsing Algorithm**) and some generates parser 
instead (like **LALR parser generator**). 

<!-- 
One benefit of `Python` is the ease of using primitive data structures.
-->

### Use parse tree or not

In some cases, a parser is only applied to generate parse trees and traversing 
of such trees is delegated to future processes, while in other cases, instant 
interpretation of partial parse trees is expected. 


#### Non-deterministic parsing

Theoretically, some grammars are to-some-level ambiguous and the intermediate or final 
output of a parser may contain more than one parse trees. Parsers which can preserve
all such parse trees at each intermediate parser state are characterized as 
**Non-deterministic Parsers**.

<!--
When the grammar defined by the user is to-some-level ambiguous, the parsing job
should either be banned by prompting ambiguity before starting the job or producing
viable parse trees (a parse forest) during and after the job. 

_Since side effects may exist when parsing with instant interpretation acoording to
the user purpose, temporal ambiguity by some parser states may cause some process 
to be performed more than once and unexpected results may occur._
--> 

In this package, a parse tree is represented as a tuple and a parse forest as a list of tuples.

Example of using non-deterministic parser, e.g. the Earley parser for ambiguous grammar
is like below:

```python
from grammar import cfg
from earley import Earley

class Gif(metaclass=cfg):

    """
    Grammar having 'dangling-else' ambiguity.
    This grammar should lead LALR parser to raise conflicts.
    """

    IF     = r'if'
    THEN   = r'then'
    ELSE   = r'else'
    EXPR   = r'e'
    SINGLE = r's'


    def stmt(ifstmt):
        return ifstmt 

    def stmt(SINGLE):
        return SINGLE 

    def ifstmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return ('ite', EXPR, stmt_1, stmt_2) 

    def ifstmt(IF, EXPR, THEN, stmt):
        return ('it', EXPR, stmt)

p = Earley(Gif)
p.parse('if e then s if e then s else s')
'''output: 
[('stmt^',
  [('stmt',
    [('ifstmt',
      ['if',
       'e',
       'then',
       ('stmt', [('ifstmt', ['if', 'e', 'then', ('stmt', ['s'])])]),
       'else',
       ('stmt', ['s'])])]),
   '\\Z']),
 ('stmt^',
  [('stmt',
    [('ifstmt',
      ['if',
       'e',
       'then',
       ('stmt',
        [('ifstmt',
          ['if', 'e', 'then', ('stmt', ['s']), 'else', ('stmt', ['s'])])])])]),
   '\\Z'])]
'''
```

#### Deterministic parsers.

Typically, a well-formed deterministic parser always produces either
one parse tree or a failure. It can handle only a subset of CFG
according to their specified characteristics.

But these grammars are always more efficient when parsing and can be
applied without generating a parse tree explicitly (commonly
generating IR code during the parsing process). Most widely used
deterministic parser is the LALR(1) parser, which is the kernel of
yacc/bison program. 

A well-designed LALR(1) parser can report conflicts clearly. 

