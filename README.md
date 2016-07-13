metaparse
=====

[Parsing][] and [Interpreting][interpreting] get done with **full power** by merely declaring **a simple Python class**<sup>[1]</sup>.

<sub>[1]. Python 3 preferred.</sub>


# Quick Example

Based on this module, syntax and semantics can be defined with class **methods**.

Given a *C*-style statement grammar in conventional [CFG][CFG] form:

```
S  →  L = R
   |  R
L  →  * R
   |  id
R  →  L
```

We can write a handy [LALR][]-parser/interpreter in Python 3 for this grammar in [SDD][]-style:

``` python
from metaparse import cfg, LALR

# Helper for translated results
ids = []

class LRVal(metaclass=cfg):

    # Special pattern ignored by the underlying tokenizer
    IGNORED = r'\s+'

    # Lexical rules with re-patterns
    EQ   = r'='
    STAR = r'\*'
    ID   = r'[_a-zA-Z]\w*'

    # Syntax-directed translation rules

    def S(L, EQ, R):
        print('Got ids:', ids)
        print('assign %s to %s' % (R, L))
        ids.clear()

    def S(R):
        return ('expr', R)

    def L(STAR, R):
        return ('REF', R)

    def L(ID):
        ids.append(ID)
        return ID

    def R(L):
        return L

P_LRVal = LALR(LRVal)
```

Using it is just easy:

``` python
>>> P_LRVal.interpret('abc')
Got ids: ['abc']
('expr', 'abc')

>>> P_LRVal.interpret('* abc = *  ** ops')
Got ids: ['abc', 'ops']
assign ('REF', ('REF', ('REF', 'ops'))) to ('REF', 'abc')
```

Tools under State-of-the-Art hardly gets more handy than this (In Python 2, there goes [a more verbose way](#python-2-compatibility)).

## Retrieving the Parse Tree

If merely the parse tree is needed rather than the semantic result, use method `parse` instead of `interpret`:

``` python
>>> tr = l_LRVal.parse('* abc = *  ** ops')
>>> tr
(S,
 [(L, [(STAR -> '*')@[0:1], (R, [(L, [(ID -> 'abc')@[2:5]])])]),
  (EQ -> '=')@[6:7],
  (R,
   [(L,
     [(STAR -> '*')@[8:9],
      (R,
       [(L,
         [(STAR -> '*')@[11:12],
          (R,
           [(L,
             [(STAR -> '*')@[12:13],
              (R, [(L, [(ID -> 'ops')@[14:17]])])])])])])])])])
```

The result is a `ParseTree` object with tuple representation. A parse leaf is just a `Token` object represented as `(<token-symbol> -> '<lexeme>')@[<position-in-input>]`.

After this, calling

```>>> tr.translate()```

delivers the same result as using `interpret` for the input.


# Design and Usage

<!--
This module provides:

- Elegant syntactic/semantic definition structure.
- Token, tokenizer, parse leaf/tree structure as well as parser interface.
- Parsing algorithms ([Earley], [GLR], [GLL], [LALR] etc.).

-->

<!-- The declaration style targets [Context-Free Grammars][CFG] with completeness check (such as detection of repeated declarations, non-reachable symbols, etc). To allow ultimate ease of use, the [BNF][BNF] grammar definition is approached by the Python `class` structure, where each method definition therein is both a **syntactic rule** associated with **semantic behavior**.
-->

The design of this module is inspired by [Parsec] in Haskell and [instaparse] in Clojure, targeting at "native parsing". It is remarkable for

* no **literal string notations** for grammar (like in [Instaparse][])
* no [DSL][] feeling<sup>[2]</sup>
* no dependencies
* no helper/intermediate files generated
* rule semantics in *pure* Python
* etc.
  <sub>[2]. may be untrue.</sub>

Though this slim module does not intend to replace more extensive tools like [ANTLR][], it is extreme handy and its integration in Python projects is seamless.

Formally, the code structure for grammar declaration with `metaparse` can be described as

``` python
from metaparse import cfg, <parser>


class <grammar-object> ( metaclass=cfg ) :

    IGNORED = <ignore-pattern>           # when not given, default pattern is r"\s"

    <terminal> = <lexeme-pattern>
    ...

    def <rule-LHS> ( <rule-RHS> ) :
        <semantic-behavior>
        ...
        return <subtree-value>

    ...


<parser> = <parser-name> ( <grammar-object> )
```

Literally, lexical rule is represented by **class attribute** assignment, whilst syntactical rule by method **signature** and semantic behavior by method **body**. In method body, the call arguments represents the value interpreted by successful parsing of subtrees.



# Further into Non-determinism

Sections above only show the *front-end* of using this module. In the *back-end*, various parsing algorithms have been/can be implemented.

`metaparse` provides `Earley` parser, which can parse any [CFG][] (currently except those with **loop**s). For exmaple, given the tricky ambiguous grammar
```
S → A B C
A → u | ε
B → E | F
C → u | ε
E → u | ε
F → u | ε
```
where `ε` denotes empty production. The corresponding `metaparse` declaration (here only syntax is concerned and semantic bodies are ignored)

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

Using Earley/GLR parser, we get all ambiguous parse trees properly:
``` python
>>> p_S = Earley(S)
>>> p_S.parse_many('u')
[(S, [(A, []), (B, [(F, [])]), (C, [(u -> 'u')@[0:1]])]),
 (S, [(A, []), (B, [(E, [])]), (C, [(u -> 'u')@[0:1]])]),
 (S, [(A, []), (B, [(F, [(u -> 'u')@[0:1]])]), (C, [])]),
 (S, [(A, []), (B, [(E, [(u -> 'u')@[0:1]])]), (C, [])]),
 (S, [(A, [(u -> 'u')@[0:1]]), (B, [(E, [])]), (C, [])]),
 (S, [(A, [(u -> 'u')@[0:1]]), (B, [(F, [])]), (C, [])])]

>>> p_S = GLR(S)
>>> p_S.parse_many('u')
[(S, [(A, [(u -> 'u')@[0:1]]), (B, [(F, [])]), (C, [])]),
 (S, [(A, [(u -> 'u')@[0:1]]), (B, [(E, [])]), (C, [])]),
 (S, [(A, []), (B, [(F, [(u -> 'u')@[0:1]])]), (C, [])]),
 (S, [(A, []), (B, [(E, [(u -> 'u')@[0:1]])]), (C, [])]),
 (S, [(A, []), (B, [(F, [])]), (C, [(u -> 'u')@[0:1]])]),
 (S, [(A, []), (B, [(E, [])]), (C, [(u -> 'u')@[0:1]])])]
```

These may be helpful for inspecting the grammar's characteristics.

Note for *non-deterministic* parsers like `Earley`, method `parse_many` should be used instead of `parse`.


<!--
## Non-determinism and ambiguity

While [LALR parser][LALR] is a classical *deterministic* parser, further parsers can be use to experiment with trickier grammars for heuristic exploration.

For example, given the famous [Dangling-Else](https://en.wikipedia.org/wiki/Dangling_else) grammar which

* is ambiguous and
* needs [left-factoring][LF] to be [LL(k)][LL].

We declare a powerful *non-deterministic* [GLL parser][Gll] to process it directly:
``` python
from metaparse import cfg, GLL

class G_IfThenElse(metaclass=cfg):

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
        # The trailing substring '_1' and '_2' denotes instances of
        # the nonterminal 'stmt' in parameter list
        return ('ite', stmt_1, stmt_2)

P_IfThenElse = GLL(G_IfThenElse)
```
and it yields multiple legal results properly:

``` python
>>> P_IfThenElse.interpret_many('if 1 then if 2 then if 3 then x else yy else zzz')
[('ite', ('ite', ('it', 'x'), 'yy'), 'zzz'),
 ('ite', ('it', ('ite', 'x', 'yy')), 'zzz'),
 ('it', ('ite', ('ite', 'x', 'yy'), 'zzz'))]
```

On the otherside, using LALR parser would report LR-conflicts due to ambiguity:
``` python
>>> from metaparse import LALR
>>> LALR(G_IfThenElse)
Traceback (most recent call last):
  File "c:/Users/Shellay/Documents/GitHub/metaparse/tests/test_if_else.py", line 117, in <module>
    LALR(Gif)
  File "c:\Users\Shellay\Documents\GitHub\metaparse\metaparse.py", line 1414, in __init__
    self._build_automaton()
  File "c:\Users\Shellay\Documents\GitHub\metaparse\metaparse.py", line 1564, in _build_automaton
    raise ParserError(msg)
metaparse.ParserError:
########## Error ##########

! LALR-Conflict raised:
  - in ACTION[7]:
{'ELSE': (SHIFT, 8), 'END': (REDUCE, (ifstmt = IF EXPR THEN stmt.))}
  * conflicting action on token 'ELSE':
{'ELSE': (REDUCE, (ifstmt = IF EXPR THEN stmt.))}
#########################
```
-->

# Limitations

Though this module provides advantageous features, there are also limitations:

* Parsing grammars with **loop**s is yet to be supported. For example, the grammar
  ```
  P → Q | a
  Q → P
  ```
  is *infinitely ambiguous*, which has infinite number of derivations while processing only finite input, e.g. `a`:
  ```
  P ⇒ a
  P ⇒ Q ⇒ P ⇒ a
  ...
  P ⇒ Q ⇒ ... ⇒ P ⇒ a
  ```
  where each derivation corresponds to a parse tree.

* No specification of operator precedence.

* Only **legal Python identifier**, rather than non-alphabetic symbols (like `<fo#o>`, `==`, `raise`, etc) can be used as symbols in grammar (seems no serious).

* Algorithms in pure Python lowers performance, but lots can be optimized.

* GLL parser is yet able to handle *left-recursion*.


# TODO-List

* Support *left-recursion* by GLL parser.

* May also support Graph-Structured-Stack for non-deterministic parsers


# Python 2 compatibility

The following version of the grammar in [the first example](#quick-example) works for both Python 2 and Python 3, relying on provided decorators `cfg2` and `rule`:

``` python
from metaparse import cfg2, rule

@cfg2
class LRVal:

    EQ   = r'='
    STAR = r'\*'
    ID   = r'[_a-zA-Z]\w*'

    @rule
    def S(L, EQ, R):
        print('Got ids:', ids)
        print('assign %s to %s' % (L, R))
        ids.clear()

    @rule
    def S(R):
        print('Got ids:', ids)
        return ('expr', R)

    @rule
    def L(STAR, R):
        return ('REF', R)
    @rule
    def L(ID):
        ids.append(ID)
        return ID

    @rule
    def R(L):
        return L
```

The problem is that `type.__prepare__` creating a method collector is not supported in Python 2, so that repeatedly declared methods can not be collected without the help of some decorator like the `rule`.

The resulted grammar instance is created by `cfg2` decorator which utilizes information collected by `rule`. Here no `metaclass` is needed.

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
[SDD]: https://en.wikipedia.org/wiki/Syntax-directed_translation "Syntax-directed Translation"
[LF]: http://www.csd.uwo.ca/~moreno//CS447/Lectures/Syntax.html/node9.html "Left-factoring"
[ANTLR]: http://www.antlr.org/ "ANother Tool for Language Recognition"

