metaparse
=====

[Parsing]: https://en.wikipedia.org/wiki/Parsing "Parsing"
[DSL]: https://en.wikipedia.org/wiki/Domain-specific_language "DSL"
[BNF]: https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_Form "BNF"
[Earley]: https://en.wikipedia.org/wiki/Earley_parser "Earley"
[LL]: https://en.wikipedia.org/wiki/LL_parser "LL"
[GLL]: http://dotat.at/tmp/gll.pdf "GLL"
[GLR]: https://en.wikipedia.org/wiki/GLR_parser "GLR"
[LALR]: https://en.wikipedia.org/wiki/LALR_parser "LALR"
[CFG]: https://en.wikipedia.org/wiki/Context-free_grammar "CFG"
[Yacc]: https://en.wikipedia.org/wiki/Yacc "Yacc"
[Bison]: https://en.wikipedia.org/wiki/GNU_bison "Bison"
[Parsec]: http://book.realworldhaskell.org/read/using-parsec.html "Parsec"
[instaparse]: https://github.com/Engelberg/instaparse "Instaparse"

[Parsing] can be done **with full power** by merely declaring **a simple Python class**[^] in lines with

* compiling just coming along. No easier.

* no **docstring notations**

* no [DSL][DSL] (maybe untrue)


<!-- > <sup>*. Python 3 preferred. Usage in Python 2 is a bit more verbose. </sup> -->
> *. Python 3 preferred. Usage in Python 2 is a bit more verbose.

## First view

Given a left-right-value grammar in a C-like language in typical [CFG][CFG] form:

```
S  →  L = R
   |  R

L  →  * R
   |  id

R  →  L
```

A handy [LALR parser][LALR] for this grammar supported by this module can be written as (in Python 3):


``` python
from metaparse import LALR

class P_LRVal(metaclass=LALR.meta):     # Use metaclass

    # Lexical elements in attribute form:
    #
    #   <lex-name> = <re-pattern>
    #
    EQ      = r'='
    STAR    = r'\*'
    ID      = r'\w+'

    # Rules in method form:
    #
    #   def <symbol> (<symbol> ... ):
    #       <do-sth>                    # Semantics in pure Python code!
    #       ...
    #       return <symbol-value>
    #
    def S(L, EQ, R) : return ('assign', L, R)
    def S(R)        : return ('expr', R)
    def L(STAR, R)  : return ('deref', R)
    def L(ID)       : return ID
    def R(L)        : return L
```

Usage is just easy:

``` python
>>> P_LRVal.interpret1('abc')
('expr', 'abc')

>>> P_LRVal.interpret1('abc = * * ops')
('assign', 'abc', ('deref', ('deref', 'ops')))

>>> P_LRVal.interpret1('* abc = * * * ops')
('assign', ('deref', 'abc'), ('deref', ('deref', ('deref', 'ops'))))
```

which seems impossible to get easier for coders to get parsing work done.


## Design

This module prepares simple and powerful utilities for syntactical analyzing and even compiling in pure Python environment. It provides:

- Elegant [BNF][BNF]-like syntactic definition through Python `class`.

- Elegant semantics definition through `def` methods.

- Token, tokenizer, parse leaf/tree structure as well as parser interface.

- Parsing algorithms ([Earley][Earley], [GLR][GLR], [GLL][GLL], [LALR][LALR] etc.).

<!-- The declaration style targets [Context-Free Grammars][CFG] with completeness check (such as detection of repeated declarations, non-reachable symbols, etc). To allow ultimate ease of use, the [BNF][BNF] grammar definition is approached by the Python `class` structure, where each method definition therein is both a **syntactic rule** associated with **semantic behavior**.
-->

The design of this module is inspired by [Parsec][Parsec] in Haskell and [Instaparse][instaparse] in Clojure, which support parsing work to be done inside programming environment (unlike traditional [Yacc][Yacc]/[Bison][Bison]-style).

## Highlights

This module is remarkable since

- Single-file module in *pure* Python
- No dependencies except built-in Python
- No intermediate files for parser generation
- No separation of declaration files
- Easy integration of Python built-in/3rd-party functionalities into rule semantics
- etc.

### Further into non-determinism

While [LALR parser][LALR] is a classical *deterministic* parser, there are further parsers which can be experimented with ambiguous grammars, which may be helpful to inspect ambiguity issues when designing a grammar.

For example, the *non-deterministic* [GLL parser][Gll] can be used to process the famous ambiguous [Dangling-Else](https://en.wikipedia.org/wiki/Dangling_else) grammar correctly. Thanks to the powerful [GLL][Gll] algorithm, there is no need for [Left-factoring](http://www.csd.uwo.ca/~moreno//CS447/Lectures/Syntax.html/node9.html), which is a serious headache of designing [LL][LL] grammars for top-down parsing (even the extremly notable [Parsec][Parsec] in Haskell cannot handle this with ease).

``` python
from metaparse import GLL

class P_IfThenElse(metaclass=GLL.meta):

    IGNORED = r'\s'
    IF      = r'if'
    THEN    = r'then'
    ELSE    = r'else'
    EXPR    = r'\d+'
    SINGLE  = r'[_a-zA-Z]\w*'

    def stmt(SINGLE):
        return SINGLE
    def stmt(IF, EXPR, THEN, stmt):
        return ('it', stmt)
    def stmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        # Symbol instances refering to the same symbol
        # are differed by subsription, here stmt_1 and
        # stmt_2
        return ('ite', stmt_1, stmt_2)
```
with multiple legally interpreted results delivered:

``` python
>>> P_IfThenElse.interpret('if 1 then if 2 then if 3 then x else yy else zzz')
[('ite', ('ite', ('it', 'x'), 'yy'), 'zzz'),
 ('ite', ('it', ('ite', 'x', 'yy')), 'zzz'),
 ('it', ('ite', ('ite', 'x', 'yy'), 'zzz'))]
```

## Explanation

It may seem unusual but interesting that such a kind of method declaration can play two roles at the same time, one is the formal syntactic rule represented by the method signature literals, while the other is the semantic behavior interpreting the rule in the Python runtime environment.


<!-- By applying the metaclass, the original behavior of Python class declaration
is overriden (this style of using metaclass is only available in Python 3.X),
which has the following new meanings:


- Attribute declarations

    - LHS is the name of the Token (lexical unit)

    - RHS is the pattern of the Token, which obeys the Python regular
    expression syntax (see documentation of the `re` module)

    - The order of declarations matters. Since there may be patterns
    that overlap, the patterns in prior positions are matched first
    during tokenizing


- Class level method declarations

  - Method name is the rule-LHS, i.e. nonterminal symbol

  - Method paramter list is the rule-RHS, i.e. a sequence of
  symbols. Moreover, each parameter binds to the successful
  subtree or result of executing the subtree's semantic rule
  during parsing the symbol

  - Method body specifies the semantic behavior associated with the
  rule. The returned value is treated as the result of successfully
  parsing input with this rule
 -->

## Limitations

Though this module supplies many advantageous features, there are also limitations:

- One limitation is that non-alphabetic word and keywords reserved for Python language are not usable as literals for declaring grammars (Similar restricltions also raise when using [Parsec][Parsec])

    - Although the representation of rule is somewhat verbose especially by the definition of alternative productions, it clearly specifies alternative semantic behaviors with NAMED parameters, which appears to be more self-descriptive than POSITIONAL parameters, like $1, $2 ... in [Yacc][Yacc]/[Bison][Bison] tool sets.

- Since the implementation of algorithms is in pure Python, performance may seem unsatisfying

<!--
This package provides (subjectively) the most simple and elegant way of getting parsing work done due to its dedicated parser front-end. It aims to be a remarkable alternative of traditional [yacc](https://en.wikipedia.org/wiki/Yacc)-style toolset in Python environment.

Summary of this toolset:

- Extreme ease of writing grammar rules
    - and rule semantics in _pure_ `Python 3`
    - enjoyment of `Python` eco-system
- No intermediate files
    - quick define, quick use
    - in-memory parser oject, serializable
- Highly portable
    - package implemented in _pure_ `Python 3` without any 3rd-party lib

### Background

The initiative for creating this package is to support studying and analyzing **[Context Free Grammars (CFG)](https://en.wikipedia.org/wiki/Context- free_grammar)** parser algorithms with the most easy parser front-end in a handy language environment like `Python`.

After the dedicated front-end is designed, various parsers like Earley, GLR(0), LALR(1) parsers can be implemented as backend.
 -->
<!--
Also LL(1) and *parsec* parser based upon Parsing Expression
Grammar(PEG) is being integrated but still not completed.
-->

<!--
## Rationale

This package is remarkable for its amazing simplicity and elegancy thanks to the extreme reflexibility of Python 3 language. It differentiates itself with the two traditional parsing tooling approaches, which IMHO yield some limitations:

- About [GNU bison](https://en.wikipedia.org/wiki/GNU_bison)

Traditional parsing work is mostly supported with the toolset flex/bison (trad. lex/yacc). Such toolset is found to be hard and complex to handle even for experienced programmers who just wants to do some handy parsing. It was also constrained in `C/C++` language environment as well as corresponding data structures and 3rd party libraries which are somewhat difficult for non-C/C++ programmers.

- About [parsec](https://wiki.haskell.org/Parsec)

Fans of functional languages like `Haskell` often appreciate the powerful **parsec** library for its expression-based translation mechanism. Pitifully, the (almost) intrinsic LL(1)-nature and explicit usage of try-function comprise significant limitations.


## Parser front-end: OO-style grammar definition

Coincidently, there is some similarity between [BNF-style](https://en.wikipedia.org/wiki/Backus–Naur_Form) grammar rule definition and `Python` function signature. Moreover, the semantic behavior corresponding to a rule can be represented by the function body.

Given a examplar rule definition in BISON:

```
exp:      NUM
        | exp exp '+'     { $$ = $1 + $2;    }
        | exp exp '-'     { $$ = $1 - $2;    }
        ...
        ;
```

the sematically equivalent representation in this package would be

```python
class RPEGrammar(metaclass=cfg):

    ...

    NUM = r'\d+'
    PLUS = r'\+'
    MINUS = r'-'

    def exp(NUM):
        return int(NUM)

    def exp(exp_1, exp_2, PLUS):
        return exp_1 + exp_2

    def exp(exp_1, exp_2, MINUS):
        return exp_1 - exp_2

    ...
```

where each nonterminal is represented as function definition and symbols in this rule's production sequence are represented by the argument list. Underscores following by a digit designate different instances of the same rule production.

## Usage: Parse trees v.s. Semantic behaviors

In this package, the user can generate parse trees when given input. Such usage is through the method `my_parser.parse(input)`. A parse tree is represented through this meta-rule within `Python` context:

```python
<parse-tree> ::= <leaf-token>                           /* :: str */
              |  (<subtree-name>, [<parse-tree> ...])   /* :: tuple[str, list] */
```

Alternatively, the user can also perform target semantics during parsing (direct translation), where the callee method is `my_parser.interpret(input)`.

### An example

While the lexical declaration can be written as _class-attribute_ declarations and the rules can be written as _method_ declarations, a grammar instance for interperting arithmetic expressions in this package can be written as follows:

```python
# Importing utilities
from grammar import cfg
from lalr import LALR

# Grammar definition
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

# Choose a parser backend for the given grammar
p = LALR(GArith)

# Produces parse tree
p.parse('3 + 2 * (5 + 11)')
# Output:
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

# Direct interpretation
p.interpret('3 + 2 * (5 + 11)')
# Output: 35
```

The key trick supporting this style is `Python 3`'s reflection and metaprogramming functionalities. By using the module `inspect`, signatures of a method can be transformed into  a pre-defined grammar rule object and the method body can be referenced as  semantic object associated with this object. By using the `metaclass` utilities, especially the `__prepare__` method, attribute and method declarations can be turned into pre-defined `Token` and `Rule` objects.


## Parser back-end: Context Free Grammar (CFG) Parsers

Based upon the concepts above, grammar representations are easily translated into grammar objects. Upon that, it is natural to prepare various parser algorithms as the back-end for various purposes. Some of them perform parsing process directly (like [Earley's parsing algorithm](https://en.wikipedia.org/wiki/Earley_parser)) whilst some generates parser instead (like [LALR parser generator](https://en.wikipedia.org/wiki/LALR_parser_generator)).

### Issues by non-deterministic parsing

Theoretically, some grammars are to-some-level ambiguous and the intermediate or final output of a parser may contain more than one parse trees. Parsers which can preserve all such parse trees at each intermediate parser state are characterized as *Non-deterministic Parsers*.


Example of using non-deterministic parser, e.g. the Earley parser for ambiguous grammar is like below:

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
# Output:
'''
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

The problem arises when the parse result needs to be interpretd at any parser state since it is not clear which parse tree should be chosen. If interpretation is performed on-the-fly during parsing, then rule semantics implemented with side-effect may lead to repeated execution of some body and unexpected result may occur.

### Deterministic parsing

A well-designed deterministic parser like *LALR* parser can report ambiguity conflicts clearly. When creating a LALR-parser with the dangling-else grammar above, a _shift/reduce_ conflict due to the ambiguity is raised:

```python
from lalr import LALR

p = LALR(Gif)

# Output:
'''
! LALR-Conflict raised:
  - in ACTION[7]:
{'ELSE': ('shift', 8), 'END': ('reduce', (ifstmt -> IF EXPR THEN stmt.))}
  * conflicting action on token 'ELSE':
    {'ELSE': ('reduce', (ifstmt -> IF EXPR THEN stmt.))}
'''
```

In such case, the generated parser structure, i.e. the underlying parser automaton states with corresponding transition tokens can be then inspected and analyzed with ease based on the OO-designed parser object.

Another popular package [PLY](http://www.dabeaz.com/ply/) supplies extensive functionalities also based on LR-parsing. That implementation has more traditional front-end rule representation and yields better performance of parsing algorithm (ca. 130% speed-up of mine according to my nonserious benchmarking). Maybe the benefits of the efficiency can be learned for future optimization of this package.
-->