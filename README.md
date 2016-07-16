metaparse
=====

You may need **instant parsing**<sup>[1]</sup> (and interpreting) with **full power**.

But, a **Python class<sup>[1]</sup> declaration** just suffices, including

* lexical definition
* rule definition
* semantic definition

all-in-one for a grammar.


<sub>[1]. This module is motivated by [instaparse][] in [Clojure][], but goes another way.</sub>
<sub>[2]. Python 3 preferred.</sub>

# Quick Example

In `metaparse`, grammar syntax and semantics can be simply defined with **class methods**.

To illustrate this, we create a tiny calculator which registers variables binding to expression results.

Firstly, we design the grammar on a paper:

```
assign → ID = expr

expr → NUM
expr → expr + expr
expr → expr * expr
expr → expr ** expr
```

then we transform this into Python form. For semantics, [SDT][]-style is used (similar to [Yacc][]).

``` python
from metaparse import cfg, LALR

# Global stuff
table = {}

class G_Calc(metaclass=cfg):

    # ===== Lexical patterns / Terminals =====

    IGNORED = r'\s+'            # Special token.

    EQ  = r'='
    NUM = r'[0-9]+'
    ID  = r'[_a-zA-Z]\w*'
    POW = r'\*\*', 3            # Can specify token precedence (mainly for LALR).
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    # ===== Syntactic/Semantic rules in SDT-style =====

    def assign(ID, EQ, expr):        # May rely on external side-effect...
        table[ID] = expr

    def expr(NUM):                   # or return local results for purity.
        return int(NUM)

    def expr(expr_1, ADD, expr_2):   # With TeX-subscripts, meaning (expr → expr₁ + expr₂).
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):     # Can ignore one of the subscripts.
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1

```

Then we build parser with this grammar object.

```
Calc = LALR(G_Calc)
```

Now we are done and it's quite easy to try it out.

``` python
>>> Calc.interpret("x = 1 + 2 * 3 ** 4 + 5")
>>> Calc.interpret("y = 3 ** 4 * 5")
>>> Calc.interpret("z = 99")

>>> table
{'x': 168, 'z': 99, 'y': 405}
```

IMO, tools in state of the art could hardly get more handy than this.

For Python 2 compatibility, there goes [another way](#python-2-compatibility)).

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

With this tree, calling ```tr.translate()``` returns the same result as using `interpret` for the input. With LALR parser, the method `interpret` performs on-the-fly interpretation without producing any parse trees explicitly (thus saves memory space).


# Design

<!--
This module provides:

- Elegant syntactic/semantic definition structure.
- Token, tokenizer, parse leaf/tree structure as well as parser interface.
- Parsing algorithms ([Earley], [GLR], [GLL], [LALR] etc.).

-->

<!-- The declaration style targets [Context-Free Grammars][CFG] with completeness check (such as detection of repeated declarations, non-reachable symbols, etc). To allow ultimate ease of use, the [BNF][BNF] grammar definition is approached by the Python `class` structure, where each method definition therein is both a **syntactic rule** associated with **semantic behavior**.
-->

The design of this module is inspired by [instaparse] in Clojure targeting at "native parsing". It is remarkable for

* native structure representing grammar rules
    - no **literal string notations** like `"E = E + T"`, `"T = F"` etc.
* rule semantics in *pure* Python
* no [DSL][] feeling<sup>[2]</sup>
* no dependencies
* no helper/intermediate files generated
* optional precedence specification (for LALR)
* etc.

thanks to [metaprogramming](https://docs.python.org/3/reference/datamodel.html#customizing-class-creation) techniques.

<sub>[2]. may be untrue.</sub>

Though this slim module does not intend to replace more extensive tools like [ANTLR][], it is extremely handy and its integration in Python projects is seamless.

# A Tiny Documentation

Demonstrated by the above example, the code structure for grammar declaration with `metaparse` can be more formally described as

``` python
from metaparse import cfg, <parser>


class <grammar-object> ( metaclass=cfg ) :

    IGNORED = <ignore-pattern>           # When not given, default pattern is r"\s".

    <terminal> = <lexeme-pattern>
    ...                                  # The order of lexical rules matters.

    def <rule-LHS> ( <rule-RHS> ) :
        <semantic-behavior>
        ...
        return <subtree-value>

    ...

<parser> = <parser-name> ( <grammar-object> )
```

Literally, lexical rule is represented by **class attribute** assignment, syntactical rule by method **signature** and semantic behavior by method **body**. In method body, the call arguments represents the values interpreted by successful parsing of subtrees.

The working mechanism of such a declaration trick is quite simple. The metaclass `cfg`

0. prepares an [assoc-list](https://en.wikipedia.org/wiki/Association_list) with [`__prepare__`](https://docs.python.org/3/reference/datamodel.html#preparing-the-class-namespace),
0. registers attributes/methods into this assoc-list,
0. in its [`__new__`](https://docs.python.org/3/reference/datamodel.html#object.__new__) translates attributes into `re` objects and methods into `Rule` objects
0. and  returns a completed `Grammar` object.

Then a `XXXParser` object can be initialized given this `Grammar` and prepares necessary table/algorithms for parsing.

# Further into Non-determinism

Instructions above only show the *front-end* of using this module. At the *back-end* side, various parsing algorithms have been/can be implemented.

Tripping further, `metaparse` provides the `Earley` parser, which can parse any [CFG][] (currently except those with **loop**s due to tree generation) as well as the `GLR` parser, which can parse LR grammars. For exmaple, given the tricky ambiguous grammar
```
S → A B C
A → u | ε
B → E | F
C → u | ε
E → u | ε
F → u | ε
```
where `ε` denotes empty production. The corresponding `metaparse` declaration (we can ignore semantic bodies here since we do not need translation)

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

These may be helpful for inspecting some grammar's characteristics.

Despite this power, `LALR` would be recommended currently for practical use since it permits *no ambiguity* (which may harm the language design but cannot be directly discovered by constructing non-deterministic parsers).

Note for *non-deterministic* parsers like `Earley` and `GLR`, method `parse_many` should be used instead of `parse` since more than one parse results may be produced.


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

* Only **legal Python identifier**, rather than non-alphabetic symbols (like `<fo#o>`, `==`, `raise`, etc) can be used as symbols in grammar (seems no serious).

* Algorithms in pure Python lowers performance, but lots can be optimized.

* GLL parser is yet able to handle *left-recursion*.


# TODO-List

* Support *left-recursion* by GLL parser.

* May also support Graph-Structured-Stack for non-deterministic parsers


# Python 2 compatibility

The following version of the grammar in [the first example](#quick-example) works for both Python 2 and Python 3, relying on provided decorators `cfg2` and `rule`:

``` python
from metaparse import cfg, LALR

@LALR
@cfg.v2
def Calc_v2():

    IGNORED = r'\s+'

    EQ  = r'='
    NUM = r'[0-9]+'
    ID  = r'[_a-zA-Z]\w*'
    POW = r'\*\*', 3
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    def assign(ID, EQ, expr):
        table[ID] = expr

    def expr(NUM):
        return int(NUM)

    def expr(expr_1, ADD, expr_2):
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1

    return
```

The problem is that `type.__prepare__` creating a method collector is not supported in Python 2. Altough `__metaclass__` is also available, it suffers from the restriction that we can *not* collect lexcial rule (Python assignment statements) in original declaration order.

Generally, unlike `class` structure, `def` structure allows deeper access to its source code through `inspect`. After some tricks with module `ast` traversing the parsed `def` function's body, the assignments can then get collected in order and methods get registered with supposedly correct specification of global/local contexts. Finally all stuff for constructing a `Grammar` object gets prepared.

Although this alternative form with merely decorators seems less verbose, it is much less explicit for understanding. Some working mechanisms may not be clear enough (especially the contexts for inner `def`s).

[clojure]: https://clojure.org/ "Clojure"
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

