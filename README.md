metaparse
=====

This is a tool for helping out **instant parsing** or **language
design** tasks with **strong power** in pure Python
environment<sup>[1]</sup>.

The biggest highlight is that preparing a Python *class*<sup>[2]</sup>
is enough to *define a language*, including

* defining lexical patterns
* defining syntatical rules
* defining semantic actions (i.e. interpretation/translation)

From such a class a parser/interpreter is automatically generated.
Based on it, parsing strings using this new language can be done by
simply calling the `parse` or `interpret` method.


<sub>[1]. This module is motivated by [instaparse][] in [Clojure][], but travels another way more like [PLY][].</sub>
<br/>
<sub>[2]. Python 3 preferred.</sub>


# Table of Contents
1. [Quick Example](#quick-example)
1. [Design and Usage](#design-and-usage)
1. [Generalized LALR Parsing](#generalized-lalr-and-dealing-with-ambiguity)
1. [API](#api)


# Quick Example

In `metaparse`, language syntax and semantics can be simply specified
with **class methods**. To illustrate this, we create a tiny
calculator grammar which can read basic arithmetic expressions and
register variable bindings in a table (aka. `context`).

At first, we conceptually design the grammar on a paper, as has been
shown in textbooks,

```
assign → ID = expr
expr → NUM
expr → ID
expr → expr₁ + expr₂
expr → expr₁ * expr₂
expr → expr₁ ** expr₂
```

then we incorporate `def` signatures in Python respectively as an alternative notation:
``` python
def assign(ID, EQ, expr): ...
def expr(NUM): ...
def expr(ID): ...
def expr(expr_1, ADD, expr_2): ...
def expr(expr_1, MUL, expr_2): ...
def expr(expr_1, POW, expr_2): ...
```

and finally we write down the semantic rules as method bodies similar
to using the [SDT][]-style (cf. [Yacc][]).

``` python
from metaparse import LALR

# Global context/environment for language semantics.
context = {}

class LangArith(metaclass=LALR.meta):

    "A language for calculating expressions."

    # ===== Lexical patterns / Terminals =====
    # - Patterns are specified via regular expressions
    # - Patterns will be checked with the same order as declared during tokenizing

    IGNORED = r'\s+'             # Special pattern to be ignored.

    EQ  = r'='
    POW = r'\*\*', 3             # Can include precedence of token using a number (for LALR conflict resolution)
    POW = r'\^'  , 3             # Alternative patterns can share the same name
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    ID  = r'[_a-zA-Z]\w*'
    NUM = r'[1-9][0-9]*'
    def NUM(value):              # Can specify translator for certain lexical patterns!
        return int(value)

    # ===== Syntactic/Semantic rules in SDT-style =====

    def assign(ID, EQ, expr):        # Can access global context in Python environment.
        context[ID] = expr
        return expr

    def expr(NUM):                   # Normally computing result without side-effects would be better.
        return NUM                   # NUM is passed as (int) since there is a NUM handler!

    def expr(ID):
        return context[ID]

    def expr(expr_1, ADD, expr_2):   # TeX style subscripts used for identifying expression instances, like (expr → expr₁ + expr₂)
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):     # Can ignore one of the subscripts.
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1
```

Then we get a `LALR` parser object:

``` python
>>> type(LangArith)
<class 'metaparse.LALR>
```

Now we are **done** and it's quite straightforward trying it out.

``` python
>>> LangArith.interpret("x = 1 + 4 * 3 ** 2 + 5")
42
>>> LangArith.interpret("y = 5 + x * 2")
89
>>> LangArith.interpret("z = 9 ^ 2")
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

The design of this module targets "**native** parsing" (like [instaparse][] and [Parsec][]). Highlights are

* native structure representing grammar rules
    - like `def E(E, plus, T)`, `def T(F)` ...
    - rather than **literal string notations** like `"E = E + T"`, `"T = F"` ...
* language translation implemented in *pure* Python,
* easy to play with (e.g. REPL),
* no generated code or program,
* no [DSL][] feeling<sup>[3]</sup>,
* support for dump/load,
* no extra dependencies,
* optional precedence specification (for LALR),
* nice error reporting,
* and etc.


<!-- All thanks to [metaprogramming](https://docs.python.org/3/reference/datamodel.html#customizing-class-creation) techniques.
 -->

<sub>[3]. may be untrue.</sub>

Though this slim module does not intend to replace full-fledged tools
like [Bison][] and [ANTLR][], it is still very handy and its
integration into existing Python project is seamless.

The following sections explains more details about the core utilities
. Feel free to skip them since you already see from above how it is
used.


## Retrieving the Parse Tree

Continuing the first example, if only the parse tree is needed rather
than the translation result, use method `parse` instead of
`interpret`:

``` python
tr = LangArith.parse(" w  = 1 + 2 * 3 ** 4 + 5 ")

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


## Save generated parser object

It can be time consuming when `metaparse` converts your language into
a parser/interpreter, depending on the size of the language. You might
not want to re-generate the parser each time you starts a Python
process.  So `metaparse` allows you to serialize your parser (which is
no much more than a dictionary encoding the state machine under the
hood). The API is `dumps/loads` or `dump/load`.

``` python
LangArith.dumps('./eg_demo_dump.py')
```

Since our parser is created given access to a global variable named
`context`, which makes `globals` and `context` dependencies of your
translation scheme, you have to pass it to `load` when loading the
parser and define the `context` object in the global scope to allow
your translation to be still functional (for sure, a better way is to
define your context object dedicatedly instead of using `globals`):

``` python
# Another file using the parser

from metaparse import LALR

# Let loaded parser be able to access current runtime env `globals()`.
arith_parser = LALR.load('./eg_demo_dump.py', globals())

# Context instance to be accessed by the loaded parser
context = {}

arith_parser.interpret('foo = 1 + 9')

print (context)
# {'foo': 10}
```

You might wonder why passing `globals` can work - It's due to that in
Python the `__code__` object can be evaluated given whatever context
and that's what `metaparse` does internally. (more basic details see
the documents for `exec` and `code` object).


## Error Reporting

During designing a language, it's very easy to make inconsistent
rules. `metaparse` provides sensible error reporting for such cases -
for example, executing the following

``` python
from metaparse import LALR

class ExprLang(metaclass=LALR.meta):

    NUM = '\d+'
    PLUS = '\+'

    def expr(expr, PLUS, term):
        return expr + term

    def expr(expr, TIMES, term):
        return expr * term

    def expr(term):
        return term

    def term(NUM):
        return int(NUM)

    def factor(NUM):
        return int(NUM)
```

would result in error report:

``` python-traceback
metaparse.LanguageError: No lexical pattern provided for terminal symbol: TIMES
- in 2th rule (expr = expr TIMES term)
- with helping traceback (if available): 
  File "test_make_error.py", line 21, in expr

- declared lexes: Lexer{
[('NUM', re.compile('\\d+')),
 ('PLUS', re.compile('\\+')),
 ('IGNORED', re.compile('\\s+'))]}
```

After providing the missing terminal symbol `TIMES`, another error is
detected during re-run:

``` python-traceback
metaparse.LanguageError: There are unreachable nonterminal at 5th rule: {'factor'}.
- with helping traceback: 
  File "test_make_error.py", line 30, in factor
```

The error information is formulated within Python *traceback* and
should be precise enough and guide you or editors to the exact place
where correction is needed.


# Generalized LALR and Dealing with Ambiguity

`metaparse` supplies an interesting extension: the `GLR` parser with
look-ahead, which can parse ambiguous grammars and help you figure out
why a grammar is ambiguous and fails to be LALR(1).

Given the famous ambiguous [Dangling-Else][] grammar:

```
 selection-statement = ...
    | IF expression THEN statement
    | IF expression THEN statement ELSE statement
```

let's build it
using `LALR`:

``` python
from metaparse import GLR, LALR

class LangIfThenElse(metaclass=LALR.meta):

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

``` python-traceback
Handling item set: 
['(ifstmt = IF EXPR THEN stmt.ELSE stmt)', '(ifstmt = IF EXPR THEN stmt.)']
Conflict on lookahead: ELSE 
- ('reduce', (ifstmt = IF EXPR THEN stmt))
- ('shift', ['(ifstmt = IF EXPR THEN stmt ELSE.stmt)'])
```

Using `GLR.meta` instead of `LALR.meta`, and `interpret_generalized` respectively:

``` python
>>> LangIfThenElse.interpret_generalized('if 1 then if 2 then if 3 then a else b else c')
[('ite', '1', ('ite', '2', ('it', '3', 'a'), 'b'), 'c'),
 ('ite', '1', ('it', '2', ('ite', '3', 'a', 'b')), 'c'),
 ('it', '1', ('ite', '2', ('ite', '3', 'a', 'b'), 'c'))]
```

the parser delivers all ambiguous parse results which cannot be
handled by LALR(1) properly. From the result you can gather more
insights about why it's ambigious.

Note that interpreting ambigious grammar is error-prone if
side-effects are involved, since the translator function for each
alternative result is executed and it is hard to understand how they
can potentially interfer. **(It is generally advised to use
side-effects-free translation when using GLR parsers!)**.


## Using Token Precedence to Resolve Conflicts

Though GLR is powerful, we may not want to keep ambiguity in practical
cases and eventually would prefer `LALR` for the sake of clarity and
performance. Very likely, ambiguity is not what you really want and
you might want to resolve ambiguity by specifying precedence of
certain tokens.

Taking the Dangling-Else example, by associate to `ELSE` a higher
precedence than `THEN` (just like the arithmetic grammar example
regarding operators), meaning when handling `stmt` between `THEN` and
`ELSE`, i.e. conflicting rules raise an `ELSE` token, the rule having
`ELSE` has higher precedence and will be chosen:

``` python
class LangIfThenElse(metaclass=LALR.meta):
    ...
    THEN = r'then', 1
    ELSE = r'else', 2
    ...
```

With this conflict resolution. The LALR parser can be constructed
successfully and parsing delivers

```
>>> LangIfThenElse.interpret('if 1 then if 2 then if 3 then a else b else c')
('it', '1', ('ite', '2', ('ite', '3', 'a', 'b'), 'c'))
```

However, in practice, precedence specification can get highly
complicated and intended behavior gets much less than explicit. It is
advised to not use precedence at all if you could find more explicit
and straightforward alternatives.


# API

The following contents give more details about the underlying utilities.

## Explicitly Registering Lexical Patterns and Syntactic Rules

The following APIs for defining the language in [the very first
example](#quick-example) works for both Python 2 and Python 3, with
the more verbose but more explicit style, heavily relying on using
decorators.

``` python
from metaparse import LALR

LangArith = LALR()

lex  = LangArith.lexer
rule = LangArith.rule

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
LangArith.make()
```

Explanation in short:

* `lex` is the `Lexer` instance associated with `LangArith`, which is also
able to collect definition of lexical patterns.

* `rule` is a decorator which extracts syntactic rule information from
the function signature and register the function itself as translator
for this rule.

## The Underlying Lexical Analyzer

After declaring the language like above, `metaparse` internally
creates a lexical analyzer as a component used by the internal parser.
Lexical analyzer maintains a list of terminal symbols of the language
defined, preserving the order they appear in the code.

``` python
>>> LangArith.lexer
Lexer{
[('IGNORED', re.compile('\\s+')),
 ('EQ', re.compile('=')),
 ('NUM', re.compile('[1-9][0-9]*')),
 ('ID', re.compile('[_a-zA-Z]\\w*')),
 ('POW', re.compile('\\*\\*')),
 ('MUL', re.compile('\\*')),
 ('ADD', re.compile('\\+'))]}
```

It runs when method `tokenize` is called and generates tokens carrying
attributes. During tokenizing, the patterns are checked respecting the
order in the list.

Note there is a pre-defined special lexical element `IGNORED`:

  * When `Lexer` reads a string matching the pattern associating
    `IGNORED`, no token is generated for the matching part of the
    string;

  * If `IGNORED` is not explicitly overriden in the user's language
    definition, it will have the default value `r'\s+'`.

We can print out the tracing of lexcial analyzing process:

``` python
>>> for token in LangArith.lexer.tokenize(" foo  = 1 + bar * 2"):
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

Moreover, it is OK to declare more lexical patterns under the same
name:

``` python
class LangArith(metaclass=LALR.meta):
    ...
    IGNORED = r' '
    IGNORED = r'\t'
    IGNORED = r'#'
    ...
    POW     = r'\*\*'
    POW     = r'\^'
    ...
```

which avoids clustering alternative sub-patterns in one `re` expression.

In practical use, you might not need to call `Lexer` at all.


## Online-Parsing behind the Scene

The `parse` and `interpret` methods are implemented internally based
on generators, which is a sort of *online-processing* behavior, i.e.

```
<get-token> —→ <process-actions> —→ <wait-for-next-token>
```

The following block of code calls the routine directly, starts it, and
traces the intermediate states:

``` python
# Prepare a parsing routine
p = LangArith.prepare()

# Start this routine
next(p)

# Send tokens one-by-one
for token in LangArith.lexer.tokenize('bar = 1 + 2 + + 3', with_end=True):
    print("Sends: ", token)
    r = p.send(token)
    print("Got:   ", r)
    print()
``` 

that is, via sending tokens to the parser one-by-one for
interpretation, an internal interpretation stack is maintained and
updated. The top element of the stack is returned wrapped in a `Just`
structure as a response to each token (which can be a reduced result
from a sequence of elements perfectly matching the rule). When token
fails processing a `ParseError` containing useful information is
returned (rather than thrown).

``` python-traceback
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


# Limitations

Though this module provides advantageous features, there are also limitations:

* Parsing grammars with **loops** is not supported. For example, the
  grammar

  ```
  P → Q | a
  Q → P
  ```

  is *infinitely ambiguous*, which has infinite number of derivations
  while processing only finite input, e.g. `"a"`:

  ```
  P ⇒ a
  P ⇒ Q ⇒ P ⇒ a
  ...
  P ⇒ Q ⇒ ... ⇒ P ⇒ a
  ```

  where each derivation corresponds to a parse tree. Eager generation
  of these trees lead to non-termination during parsing.

* Only **legal Python identifier**, rather than non-alphabetic symbols
  (like `<fo#o>`, `==`, `raise`, etc) can be used as symbols in
  grammar (seems no serious).

* Parsing algorithms are implemented in pure Python, but speed-up via
  Cython should be possible in the future.


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
