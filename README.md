metaparse
=====

This is a tool for **instant parsing** with **full power** in native Python environment<sup>[1]</sup>.

Moreover, You might be amazed that defining a Python *class*<sup>[2]</sup> just suffices to *define a language*, including

* lexical
* syntatical
* semantic

definitions altogether, as well as the parser for it.

Then parsing work gets done by simply calling its `parse` method.


<sub>[1]. This module is motivated by [instaparse][] in [Clojure][], but travels another way more like [PLY][].</sub>
<br/>
<sub>[2]. Python 3 preferred.</sub>


# Table of Contents
1. [Quick Example](#quick-example)
1. [Design and Usage](#design-and-usage)
1. [Generalized LALR Parsing](#generalized-lalr-and-dealing-with-ambiguity)
1. [API](#api)


# Quick Example

In `metaparse`, language syntax and semantics can be simply defined
with **class methods**. To illustrate this, we create a tiny
calculator which can read basic arithmetic expressions and register
variable bindings in a table (aka. `context`).

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
from metaparse import LALR

# Global context/environment for language semantics.
context = {}

class pCalc(metaclass=LALR.meta):

    "A language for calculating expressions."

    # ===== Lexical patterns / Terminals =====
    # - Patterns specified with regular expressions
    # - Patterns will be tested in declaration order during tokenizing

    IGNORED = r'\s+'             # Special pattern to be ignored.

    EQ  = r'='
    POW = r'\*\*', 3             # Can specify precedence of token (for LALR conflict resolution)
    POW = r'\^'  , 3             # Alternative patterns can share the same name
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    ID  = r'[_a-zA-Z]\w*'
    NUM = r'[1-9][0-9]*'
    def NUM(value):              # Can specify handler for lexical pattern!
        return int(value)

    # ===== Syntactic/Semantic rules in SDT-style =====

    def assign(ID, EQ, expr):        # May access global context.
        context[ID] = expr
        return expr

    def expr(NUM):                   # May compute result purely.
        return NUM                   # NUM is passed as (int) due to the handler!

    def expr(ID):
        return context[ID]

    def expr(expr_1, ADD, expr_2):   # With TeX-subscripts, meaning (expr → expr₁ + expr₂)
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):     # Can ignore one of the subscripts.
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1
```

Then we get a `LALR` parser object:

``` python
>>> type(pCalc)
<class 'metaparse.LALR>
```

Now we are **done** and it's quite easy to try it out.

``` python
>>> pCalc.interpret("x = 1 + 4 * 3 ** 2 + 5")
42
>>> pCalc.interpret("y = 5 + x * 2")
89
>>> pCalc.interpret("z = 9 ^ 2")
81

>>> context
{'y': 89, 'x': 42, 'z': 81}
```

IMO, tools under state-of-the-art could hardly get more handy than
this.

Note `metaclass=LALR.meta` only works in Python 3. There is an
[alternative](#verbose-style) form which also works in Python 2, as
well expose the API of this module more explicitly.


# Design and Usage

The design of this module targets "**native** parsing" (like [instaparse][] and [Parsec][]). Users might find `metaparse` remarkable due to its

* native structure representing grammar rules
    - like `def E(E, plus, T)`, `def T(F)` ...
    - rather than **literal string notations** like `"E = E + T"`, `"T = F"` ...
* language semantics in *pure* Python,
* easy to play with (e.g. REPL),
* no [DSL][] feeling<sup>[3]</sup>,
* dump/load utilities,
* no dependencies,
* no helper/intermediate files generated,
* optional precedence specification (for LALR),
* and etc.


<!-- All thanks to [metaprogramming](https://docs.python.org/3/reference/datamodel.html#customizing-class-creation) techniques.
 -->

<sub>[3]. may be untrue.</sub>

Though this slim module does not intend to replace more extensive
tools like [Bison][] and [ANTLR][], it is extremely handy and its
integration in Python projects can be seamless.

The following contents are unnecessary for using this module with
dirty hands, but gives better explanation about core utilities of this
module.


## Retrieving the Parse Tree

Continuing the first example, if merely the parse tree is needed
rather than the semantic result, use method `parse` instead of
`interpret`:

``` python
tr = pCalc.parse(" w  = 1 + 2 * 3 ** 4 + 5 ")

>>> tr
('assign',
 [('ID', 'w'),
  ('EQ', '='),
  ('expr',
   [('expr',
     [('expr', [('NUM', '1')]),
      ('ADD', '+'),
      ('expr',
       [('expr', [('NUM', '2')]),
        ('MUL', '*'),
        ('expr',
         [('expr', [('NUM', '3')]),
          ('POW', '**'),
          ('expr', [('NUM', '4')])])])]),
    ('ADD', '+'),
    ('expr', [('NUM', '5')])])])
```

The result is a `ParseTree` object with tuple representation. A parse
leaf is just a `Token` object represented as ```(<token-name>,
'<lexeme>')```.


## Save with Persistence

It should be useful to save the created parser persistently for future
use (since creating a new LALR parser instance may be time consuming
in some cases). Using `dumps/loads` or `dump/load` the parser instance
(more precisely, the underlying automaton) will be encoded into
*Python structure form*.

``` python
pCalc.dumps('./eg_demo_dump.py')
```

Since our parser is created given access to a global variable named
`context`, we should provide the runtime environment in the user
module, i.e. `globals`, when loading this parser instance:

``` python
# Another file using the parser

from metaparse import LALR

# Let loaded parser be able to access current runtime env `globals()`.
qCalc = LALR.load('./eg_demo_dump.py', globals())

# Context instance to be accessed by the loaded parser
context = {}

qCalc.interpret('foo = 1 + 9')

print (context)
# {'foo': 10}
```

Since the runtime environment in Python is simply a `dict`, more
tricks would be possible by passing a user-defined runtime environment
instance to the `load` method - it would be treated as the context for
executing the semantic `__code__` object (more basic details see the
documents for `exec` and `code` object).


# Generalized LALR and Dealing with Ambiguity

`metaparse` supplies an interesting extension: the `GLR` parser with look-ahead, which can cope with many non-singular ambiguous grammars.

Given the famous ambiguous [Dangling-Else][] grammar, trying to build it using `LALR`:

``` python
from metaparse import GLR, LALR

class pIfThenElse(metaclass=GLR.meta):

    IF     = r'if'
    THEN   = r'then'
    ELSE   = r'else'
    EXPR   = r'\d+'
    SINGLE = r'[_a-zA-Z]+'

    def stmt(ifstmt):
        return ifstmt 

    def stmt(SINGLE):
        return SINGLE 

    def ifstmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return ('ite', EXPR, stmt_1, stmt_2) 

    def ifstmt(IF, EXPR, THEN, stmt):
        return ('it', EXPR, stmt)
```

would result in a *shift/reduce* conflict on the token `ELSE` with error hints:

``` python
metaparse.Error: 
Handling item set: 
['(ifstmt = IF EXPR THEN stmt.ELSE stmt)', '(ifstmt = IF EXPR THEN stmt.)']
Conflict on lookahead: ELSE 
- ('reduce', (ifstmt = IF EXPR THEN stmt))
- ('shift', ['(ifstmt = IF EXPR THEN stmt ELSE.stmt)'])
```

Using `GLR.meta` instead of `LALR.meta`, and `interpret_generalized` respectively:

```
>>> pIfThenElse.interpret_generalized('if 1 then if 2 then if 3 then a else b else c')
[('ite', '1', ('ite', '2', ('it', '3', 'a'), 'b'), 'c'),
 ('ite', '1', ('it', '2', ('ite', '3', 'a', 'b')), 'c'),
 ('it', '1', ('ite', '2', ('ite', '3', 'a', 'b'), 'c'))]
```

the parser delivers all ambiguous parse results properly. Note that ambiguous parsing relying on *interpret on-the-fly* is dangerous since some latterly rejected parses might already get interpreted producing side-effects **(so it is advised to use side-effects-free semantics when using GLR parsers!)**.


## Using Precedence to Resolve Conflicts

Though GLR is quite powerful, we may not want to deal with ambiguity in practical cases and prefer `LALR` for its clarity and performance. 

By specifying `ELSE` a higher associative precedence than `THEN` (just like the calculator example treating operators), meaning `ELSE` would be preferred to combine the nearest `THEN` leftwards:

``` python
class pIfThenElse(metaclass=LALR.meta):
    ...
    THEN = r'then', 1
    ELSE = r'else', 2
    ...
```

we then get rid of ambiguity. The successful LALR parser delivers

```
>>> pIfThenElse.interpret('if 1 then if 2 then if 3 then a else b else c')
('it', '1', ('ite', '2', ('ite', '3', 'a', 'b'), 'c'))
```

However, rather than the examples here, precedence specification can be highly complex and involving in practical cases.


# API

The following contents give more details about the underlying utilities.

## Explicitly Registering Lexical Patterns and Syntactic Rules

The following version of declaring the language in
[the first example](#quick-example) works for both Python 2 and Python
3, with the more verbose but more explicit style, heavily relying on
using decorators.

``` python
from metaparse import LALR

pCalc = LALR()

lex  = pCalc.lexer
rule = pCalc.rule

# lex(<terminal-symbol> = <pattern>)
lex(IGNORED = r'\s+')
lex(NUM = r'[0-9]+')
lex(EQ  = r'=')
lex(ID  = r'[_a-zA-Z]\w*')

# lex(... , p = <precedence>)
lex(POW = r'\*\*', p=3)
lex(POW = r'\^')                # No need to give the precedence twice for POW.
lex(MUL = r'\*'  , p=2)
lex(ADD = r'\+'  , p=1)

# @rule
# def <lhs> ( <rhs> ):
#     <semantics>
@rule
def assign(ID, EQ, expr):
    context[ID] = expr
    return expr

@rule
def expr(ID):
    return context[ID]

@rule
def expr(NUM):
    return int(NUM)

@rule
def expr(expr_1, ADD, expr_2):
    return expr_1 + expr_2

@rule
def expr(expr, MUL, expr_1):
    return expr * expr_1

@rule
def expr(expr, POW, expr_1):
    return expr ** expr_1

# Complete making the parser after collecting things!
pCalc.make()
```

Explanation in short:

* `lex` is the `Lexer` instance associated with `pCalc`, which is also
able to collect definition of lexical patterns.

* `rule` is a decorator which extracted syntactic rule information
from the function signature and register this rule, as well as
register this function as semantics associating this rule.

## The Underlying Lexical Analyzer

After declaring the language like above, a lexical analyzer is created
as a utility for the usable parser, which maintains a list of terminal
symbols of the language defined, preserving their *appearance order*
in the declaration.

``` python
>>> pCalc.lexer
Lexer{
[('IGNORED', re.compile('\\s+')),
 ('EQ', re.compile('=')),
 ('NUM', re.compile('[1-9][0-9]*')),
 ('ID', re.compile('[_a-zA-Z]\\w*')),
 ('POW', re.compile('\\*\\*')),
 ('MUL', re.compile('\\*')),
 ('ADD', re.compile('\\+'))]}
```

It works by calling `tokenize` and generates tokens with informative
attributes. During tokenizing, the patterns are tested for matching
with respect to the list order.

Note there is a special lexical element `IGNORED`:

  * When `Lexer` reads a string prefix matching the pattern
    associating `IGNORED`, no token is generated for such string
    prefix;

  * If `IGNORED` is not given explicitly in the user's language
    declaration, it would be given default pattern `r'\s+'`.

We can print out the tracing of lexcial analyzing process:

``` python
>>> for token in pCalc.lexer.tokenize(" foo  = 1 + bar * 2"):
...     print(token.pos,
...           token.end,
...           token.symbol,
...           repr(token.lexeme),   # (lexeme) is something literal.
...           repr(token.value))    # (value) is something computed by handler, if exists.

1 4 ID 'foo' 'foo'
6 7 EQ '=' '='
8 9 NUM '1' 1
10 11 ADD '+' '+'
12 15 ID 'bar' 'bar'
16 17 MUL '*' '*'
18 19 NUM '2' 2

```

Moreover, it is no problem to declare more lexical patterns with the
same name, which may occasionally be useful like

``` python
class pCalc(metaclass=LALR.meta):
    ...
    IGNORED = r' '
    IGNORED = r'\t'
    IGNORED = r'#'
    ...
    POW     = r'\*\*'
    POW     = r'\^'
    ...
```

avoiding clustering alternative sub-patterns in one `re` pattern.

You often totally need no access to the `Lexer` object to use the
parser. This section only serve to be informative.


## Online-Parsing behind the Scene

The `parse` and `interpret` methods for outmost usage are based on the
more subtle parsing **coroutine**, which is designed with *online*
behavior, i.e.

```
<get-token> —→ <process-actions> —→ <wait-for-next-token>
```

The following block of code calls the routine directly, starts it, and
traces the intermediate states:

``` python
# Prepare a parsing routine
p = pCalc.prepare()

# Start this routine
next(p)

# Send tokens one-by-one
for token in pCalc.lexer.tokenize('bar = 1 + 2 + + 3', with_end=True):
    print("Sends: ", token)
    r = p.send(token)
    print("Got:   ", r)
    print()
``` 

we get the following output, where successful each intermediate result
is wrapped by `Just` and failure reported by `ParseError` containing
tracing information (returned rather than thrown).

```
Sends:  ('ID', 'bar')
Got:    Just(result=('ID', 'bar'))

Sends:  ('EQ', '=')
Got:    Just(result=('EQ', '='))

Sends:  ('NUM', '1')
Got:    Just(result=('NUM', '1'))

Sends:  ('ADD', '+')
Got:    Just(result=('ADD', '+'))

Sends:  ('NUM', '2')
Got:    Just(result=('NUM', '2'))

Sends:  ('ADD', '+')
Got:    Just(result=('ADD', '+'))

Sends:  ('ADD', '+')
Got:    Unexpected token ('ADD', '+') at (14:15)
while expecting actions 
{'ID': ('shift', 5), 'NUM': ('shift', 6)}
with state stack 
[['(assign^ = .assign)'],
 ['(assign = ID.EQ expr)'],
 ['(assign = ID EQ.expr)'],
 ['(assign = ID EQ expr.)',
  '(expr = expr.ADD expr)',
  '(expr = expr.MUL expr)',
  '(expr = expr.POW expr)'],
 ['(expr = expr ADD.expr)']]
and subtree stack 
['bar', '=', 3, '+']


Sends:  ('NUM', '3')
Got:    Just(result=('NUM', '3'))

Sends:  ('\x03', None)
Got:    Just(result=6)
```

The structured and detailed error reporting should be useful for
applications potentially built on this API.

<!-- These utilities should supply a good API for buiding applications -->
<!-- dealing with grammars supported by extensive error reporting. -->


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

* Algorithms in pure Python lowers performance, but speed-up is possible.


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
