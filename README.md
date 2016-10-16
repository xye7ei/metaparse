metaparse
=====

This is a tool for **instant parsing** with **full power** in native Python environment<sup>[1]</sup>.

Moreover, You might be amazed that merely defining a Python `class`<sup>[2]</sup> just suffices to get parse work done, including

* lexical
* syntatical
* semantic

definitions altogether.


<sub>[1]. This module is motivated by [instaparse][] in [Clojure][], but travels another way more like [PLY][].</sub>

<sub>[2]. Python 3 preferred.</sub>

# Quick Example

In `metaparse`, grammar syntax and semantics can be simply defined with **class methods**. To illustrate this, we create a tiny calculator which can do basic arithmetics and registers variable bindings in a table.

Firstly, we design the grammar on a paper, as in textbooks,

```
assign → ID = expr
expr → NUM
expr → ID
expr → expr₁ + expr₂
expr → expr₁ * expr₂
expr → expr₁ ** expr₂
```

then think about some similarity with `def` signatures in Python:
``` python
def assign(ID, EQ, expr): ...
def expr(NUM): ...
def expr(ID): ...
def expr(expr_1, ADD, expr_2): ...
def expr(expr_1, MUL, expr_2): ...
def expr(expr_1, POW, expr_2): ...
```

and finally write down the semantics with [SDT][]-style (cf. [Yacc][]).

``` python
from metaparse import cfg, LALR

# Global stuff
table = {}

class G_Calc(metaclass=cfg):

    # ===== Lexical patterns / Terminals =====
    # - Will be matched in order when tokenizing

    IGNORED = r'\s+'             # Special token.

    EQ  = r'='
    NUM = r'[1-9][0-9]*'
    ID  = r'[_a-zA-Z]\w*'
    POW = r'\*\*', 3             # Can specify precedence of token (mainly for LALR)
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    # ===== Syntactic/Semantic rules in SDT-style =====

    def assign(ID, EQ, expr):        # May rely on side-effect...
        table[ID] = expr
        return expr

    def expr(NUM):                   # or return local results for purity
        return int(NUM)

    def expr(ID):
        return table[ID]

    def expr(expr_1, ADD, expr_2):   # With TeX-subscripts, meaning (expr → expr₁ + expr₂)
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):     # Can ignore one of the subscripts.
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1
```

Then we get a `Grammar` object and build parser with it:

``` python
>>> type(G_Calc)
<class 'metaparse.Grammar'>
>>> pCalc = LALR(G_Calc)
```


Now we are done and it's quite easy to try it out.

``` python
>>> pCalc.interpret("x = 1 + 4 * 3 ** 2 + 5")
42
>>> pCalc.interpret("y = 5 + x * 2")
89
>>> pCalc.interpret("z = 99")
99

>>> table
{'x': 42, 'y': 89, 'z': 99}
```

IMO, tools in state-of-the-art could hardly get more handy than this.

Note `metaclass=cfg` only works in Python 3. There is an [alternative](#verbose-style) form which also works in Python 2 but seems trickier and is arguably less recommended<sup>[3]</sup> .

<sub>[3]. although more interesting.</sub>

## Retrieving the Parse Tree

If merely the parse tree is needed rather than the semantic result, use method `parse` instead of `interpret`:

``` python
>>> tr = Calc.parse(" w  = 1 + 2 * 3 ** 4 + 5 ")
>>> tr
(assign,
 [(ID: 'w')@[1:2],
  (EQ: '=')@[4:5],
  (expr,
   [(expr,
     [(expr, [(NUM: '1')@[6:7]]),
      (ADD: '+')@[8:9],
      (expr,
       [(expr, [(NUM: '2')@[10:11]]),
        (MUL: '*')@[12:13],
        (expr,
         [(expr, [(NUM: '3')@[14:15]]),
          (POW: '**')@[16:18],
          (expr, [(NUM: '4')@[19:20]])])])]),
    (ADD: '+')@[21:22],
    (expr, [(NUM: '5')@[23:24]])])])
```

The result is a `ParseTree` object with tuple representation. A parse leaf is just a `Token` object represented as ```(<token-name>: '<lexeme>')@[<position-in-input>]```.

Having this tree, calling ```tr.translate()``` returns the same result as `interpret`. With LALR parser, the method `interpret` performs on-the-fly interpretation without producing any parse tree explicitly.


# Design

<!--
This module provides:

- Elegant syntactic/semantic definition structure.
- Token, tokenizer, parse leaf/tree structure as well as parser interface.
- Parsing algorithms ([Earley], [GLR], [GLL], [LALR] etc.).

-->

<!-- The declaration style targets [Context-Free Grammars][CFG] with completeness check (such as detection of repeated declarations, non-reachable symbols, etc). To allow ultimate ease of use, the [BNF][BNF] grammar definition is approached by the Python `class` structure, where each method definition therein is both a **syntactic rule** associated with **semantic behavior**.
-->

The design of this module targets "native parsing" (like [instaparse][] and [Parsec][]). Users might find `metaparse` remarkable, since

* native structure representing grammar rules,
    - no **literal string notations** like `"E = E + T"`, `"T = F"` ...
* rule semantics in *pure* Python,
* easy to play with (like in REPL),
* no [DSL][] feeling<sup>[4]</sup>,
* no dependencies,
* no helper/intermediate files generated,
* optional precedence specification (for LALR),
* and etc.


<!-- All thanks to [metaprogramming](https://docs.python.org/3/reference/datamodel.html#customizing-class-creation) techniques.
 -->

<sub>[4]. may be untrue.</sub>

Though this slim module does not intend to replace more extensive tools like [Bison][] and [ANTLR][], it is extremely handy and its integration in Python projects can be seamless.


# A Tiny Documentation

Demonstrated by the above example, the code structure for grammar declaration with `metaparse` can be more formally described as

``` python
from metaparse import cfg, <parser>


class <grammar-object> ( metaclass=cfg ) :

    IGNORED = <ignored-pattern>           # When not given, default pattern is r"\s".

    <terminal> = <lexeme-pattern>
    ...                                   # The order of lexical rules matters.

    def <rule-LHS> ( <rule-RHS> ) :
        <semantic-behavior>
        ...
        return <subtree-value>

    ...

my_parser = <parser> ( <grammar-object> )
```

Literally, lexical rule is represented by **class attribute** assignment, syntactical rule by method **signature** and semantic behavior by method **body**. In method body, the call arguments represents the values interpreted by successful parsing of subtrees.

It is advised not to declare grammar-dependent attributes in such a class as helpers due to the speccial meanings.

<!--
The working mechanism of such a declaration trick is quite simple. The metaclass `cfg`

0. prepares an [assoc-list](https://en.wikipedia.org/wiki/Association_list) with [`__prepare__`](https://docs.python.org/3/reference/datamodel.html#preparing-the-class-namespace),
0. registers attributes/methods into this assoc-list,
0. in its [`__new__`](https://docs.python.org/3/reference/datamodel.html#object.__new__) translates attributes into `re` objects and methods into `Rule` objects
0. and  returns a completed `Grammar` object.

Then a `XXXParser` object can be initialized given this `Grammar` and prepares necessary table/algorithms for parsing.

-->

# Further into Non-determinism

Contents above only show the *front-end* of applying this module. On the *back-end* side, various parsing algorithms have been/can be implemented.

One step further, `metaparse` provides implementation of *non-deterministic* parsers like [`Earley`][Earley] and [`GLR`][GLR]. Currently, grammars with **loop**s are yet supported due to the eager mechanism of tree generation.

For exmaple, given the tricky *ambiguous* grammar
```
S → A B C
A → u | ε
B → E | F
C → u | ε
E → u | ε
F → u | ε
```
where `ε` denotes empty production. The corresponding `metaparse` declaration follows (we can ignore semantic bodies since we do not need translation)

``` python
from metaparse import cfg, Earley, GLR

class S(metaclass=cfg):
    u = r'u'
    def S(A, B, C) : pass
    def A(u)       : pass
    def A()        : pass
    def B(E)       : pass
    def B(F)       : pass
    def C(u)       : pass
    def C()        : pass
    def E(u)       : pass
    def E()        : pass
    def F(u)       : pass
    def F()        : pass
```

and using Earley/GLR parser, we get all *ambiguous* parse trees properly:
``` python
>>> p_S = Earley(S)
>>> p_S.parse_many('u')
[(S, [(A, []), (B, [(F, [])]), (C, [(u: 'u')@[0:1]])]),
 (S, [(A, []), (B, [(E, [])]), (C, [(u: 'u')@[0:1]])]),
 (S, [(A, []), (B, [(F, [(u: 'u')@[0:1]])]), (C, [])]),
 (S, [(A, []), (B, [(E, [(u: 'u')@[0:1]])]), (C, [])]),
 (S, [(A, [(u: 'u')@[0:1]]), (B, [(E, [])]), (C, [])]),
 (S, [(A, [(u: 'u')@[0:1]]), (B, [(F, [])]), (C, [])])]

>>> p_S = GLR(S)
>>> p_S.parse_many('u')
[(S, [(A, [(u: 'u')@[0:1]]), (B, [(F, [])]), (C, [])]),
 (S, [(A, [(u: 'u')@[0:1]]), (B, [(E, [])]), (C, [])]),
 (S, [(A, []), (B, [(F, [(u: 'u')@[0:1]])]), (C, [])]),
 (S, [(A, []), (B, [(E, [(u: 'u')@[0:1]])]), (C, [])]),
 (S, [(A, []), (B, [(F, [])]), (C, [(u: 'u')@[0:1]])]),
 (S, [(A, []), (B, [(E, [])]), (C, [(u: 'u')@[0:1]])])]
```

These results may be helpful for inspecting the grammar's characteristics.

Despite this power of non-determinism, `LALR` would be recommended currently for practical use since it won't permit *ambiguity* (which may harm the language design but cannot be directly discovered by constructing non-deterministic parsers).

Note for non-deterministic parsers like `Earley` and `GLR`, method `parse_many` should be used instead of `parse` since more than one parse results may be produced.

### Play with Ambiguity

Here is another example, showing the ambiguity of the famous [Dangling-Else][] grammar
``` python
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
```

with exemplar semantic results:

```
>>> p_ite = GLR(G_IfThenElse)
>>> p_ite.interpret_many('if 1 then if 2 then if 3 then a else b else c')
[('i-t', ('i-t-e', ('i-t-e', 'a', 'b'), 'c')),
 ('i-t-e', ('i-t', ('i-t-e', 'a', 'b')), 'c'),
 ('i-t-e', ('i-t-e', ('i-t', 'a'), 'b'), 'c')]
```

This may fail when try to build an LALR:

``` python
>>> p_ite = LALR(G_IfThenElse)
Traceback (most recent call last):
  File ...
  ...
  File "c:\Users\Shellay\Documents\GitHub\metaparse\metaparse.py", line 1801, in _build_automaton
    raise ParserError(msg)metaparse.ParserError:
============================
! LALR-Conflict raised:
  * in state [6]:
[(stmt = IF EXPR THEN stmt.), (stmt = IF EXPR THEN stmt.ELSE stmt)]
  * on lookahead 'ELSE':
{'ELSE': [(SHIFT, 7), (REDUCE, (stmt = IF EXPR THEN stmt.))]}
============================
```

However, by specifying `ELSE` a higher associative precedence than `THEN` (just like the calculator example treating operators)

``` python
class G_IfThenElse(metaclass=cfg):
    ...
    THEN = r'then', 1
    ELSE = r'else', 2
    ...
```

we then get rid of ambiguity. The successful LALR delivers
```
>>> p_lalr_ite = LALR(G_IfThenElse)
>>> p_lalr_ite.interpret('if 1 then if 2 then if 3 then a else b else c')
('i-t', ('i-t-e', ('i-t-e', 'a', 'b'), 'c'))
```

In practice, rather than the examples here, precedence specification can be highly complex and involving.


# Limitations

Though this module provides advantageous features, there are also limitations:

* Parsing grammars with **loop**s is yet to be supported. For example, the grammar
  ```
  P → Q | a
  Q → P
  ```
  is *infinitely ambiguous*, which has infinite number of derivations while processing only finite input, e.g. `"a"`:
  ```
  P ⇒ a
  P ⇒ Q ⇒ P ⇒ a
  ...
  P ⇒ Q ⇒ ... ⇒ P ⇒ a
  ```
  where each derivation corresponds to a parse tree. Eager generation of these trees lead to non-termination during parsing.

* Only **legal Python identifier**, rather than non-alphabetic symbols (like `<fo#o>`, `==`, `raise`, etc) can be used as symbols in grammar (seems no serious).

* Algorithms in pure Python lowers performance, but lots can be optimized.

* GLL parser is yet able to handle *left-recursion*.


# TODOs

* Support Graph-Structured-Stack (GSS) for non-deterministic parsers.

* Support *left-recursion* by GLL parser.


# Verbose style

The following version of the grammar in [the first example](#quick-example) works for both Python 2 and Python 3, with more verbose but more straightforward style.

``` python
from metaparse import verbose, LALR

@verbose
def Calc(lex, rule):

    lex(IGNORED = r'\s+')

    @lex(NUM = r'[0-9]+')
    def NUM(val):
        return int(val)

    lex(EQ  = r'=')
    lex(ID  = r'[_a-zA-Z]\w*')

    lex(POW = r'\*\*', p=3)
    lex(MUL = r'\*'  , p=2)
    lex(ADD = r'\+'  , p=1)
    lex(SUB = r'\-'  , p=1)

    @rule
    def assign(ID, EQ, expr):
        table[ID] = expr
        return expr

    @rule
    def expr(ID):
        return table[ID]

    @rule
    def expr(NUM):
        return int(NUM)

    @rule
    def expr(expr_1, ADD, expr_2):
        return expr_1 + expr_2

    @rule
    def expr(expr_1, SUB, expr_2):
        return expr_1 - expr_2

    @rule
    def expr(expr, MUL, expr_1):
        return expr * expr_1

    @rule
    def expr(expr, POW, expr_1):
        return expr ** expr_1


pCalc = LALR(Calc)
```

Such style would be preferred by explicity lovers, although typing the decorators `lex` and `rule` repeatedly may be sort of nuisance. The underlying mechanism is quite easy: the decorator `verbose` prepares a lex-collector and a rule-collector to be arguments passed to function `Calc`, after calling of which the lexical and syntactic rules collected are then used to construct a `Grammar` object, which gets returned with name `Calc`.


[Parsing]: https://en.wikipedia.org/wiki/Parsing "Parsing"
[Interpreting]: https://en.wikipedia.org/wiki/Interpreter_(computing) "Interpreter"
[DSL]: https://en.wikipedia.org/wiki/Domain-specific_language "Domain-specific Language"
[BNF]: https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_Form "Backus-Naur From"
[Earley]: https://en.wikipedia.org/wiki/Earley_parser "Earley"
[LL]: https://en.wikipedia.org/wiki/LL_parser "Left-to-right, Leftmost-derivation"
[GLL]: http://dotat.at/tmp/gll.pdf "General Left-to-right, Leftmost-derivation"
[GLR]: https://en.wikipedia.org/wiki/GLR_parser "General Left-to-right, Rightmost derivation"
[LALR]: https://en.wikipedia.org/wiki/LALR_parser "Look-Ahead Left-to-right, Rightmost-derivation"
[CFG]: https://en.wikipedia.org/wiki/Context-free_grammar "Context-free Grammar"
[Yacc]: https://en.wikipedia.org/wiki/Yacc "Yet Another Compiler Compiler"
[Bison]: https://en.wikipedia.org/wiki/GNU_bison "Bison"
[Parsec]: http://book.realworldhaskell.org/read/using-parsec.html "Parsec"
[instaparse]: https://github.com/Engelberg/instaparse "Instaparse"
[SDT]: https://en.wikipedia.org/wiki/Syntax-directed_translation "Syntax-directed Translation"
[LF]: http://www.csd.uwo.ca/~moreno//CS447/Lectures/Syntax.html/node9.html "Left-factoring"
[ANTLR]: http://www.antlr.org/ "ANother Tool for Language Recognition"
[clojure]: https://clojure.org/ "Clojure"
[PLY]: http://www.dabeaz.com/ply/ "PLY"
[Dangling-Else]: https://en.wikipedia.org/wiki/Dangling_else "Dangling-Else"
