metaparse
=====

[Parsing][] can be done **with full power** by merely declaring **a simple Python class**<sup>[1]</sup> and semantics for compiling just comes along.

<sub>[1]. Python 3 preferred however.</sub>

## Quick Example

Given a left-right-value grammar in a C-like language in conventional [CFG][CFG] form:

```
S  →  L = R
   |  R
L  →  * R
   |  id
R  →  L
```

A handy [LALR]-parser/translator for this grammar supported by this module can be written in [SDD]-style as:

``` python
from metaparse import cfg, LALR

# Helper for translated results
ids = []

class LRVal(metaclass=cfg):    # Using Python 3 metaclass

    # Lexical rules
    EQ   = r'='
    STAR = r'\*'
    ID   = r'[_a-zA-Z]\w*'


    # Syntax-directed translation rules

    def S(L, EQ, R):
        print('Got identifiers:', ids)
        print('assign %s to %s' % (L, R))
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

# Or alternatively, 
#
# class P_LRVal(metaclass=LALR.meta):
#     <same-stuff> ...
```

Usage is just easy:

``` python
>>> P_LRVal.interpret1('abc')
Got ids: ['abc']
('expr', 'abc')

>>> P_LRVal.interpret1('* abc = *  ** ops')
Got ids: ['abc', 'ops']
assign ('REF', 'abc') to ('REF', ('REF', ('REF', 'ops')))
```

Tools under State-of-the-Art can hardly get more handy and expressive than this. In Python 2 there goes [another way](#python-2-compatibility).


## Design

<!-- 
This module provides:

- Elegant syntactic/semantic definition structure.
- Token, tokenizer, parse leaf/tree structure as well as parser interface.
- Parsing algorithms ([Earley], [GLR], [GLL], [LALR] etc.).

 -->
 <!-- The declaration style targets [Context-Free Grammars][CFG] with completeness check (such as detection of repeated declarations, non-reachable symbols, etc). To allow ultimate ease of use, the [BNF][BNF] grammar definition is approached by the Python `class` structure, where each method definition therein is both a **syntactic rule** associated with **semantic behavior**.
-->

The design of this module is inspired by [Parsec] in Haskell and [instaparse] in Clojure, targeting at "native" parsing. It is remarkable for

<!-- - *Pure* Python -->
* no **string notations** for grammar (like in [Instaparse][]) and
* no [DSL][] <sup>[2]</sup>
* **no** dependencies
* **no** helper/intermediate files when using
* rule semantics in *pure* Python

<sub>[2]. Fakingly.</sub>

etc. Integration of this module in Python projects is seemless.

## Into non-determinism

While [LALR parser][LALR] is a classical *deterministic* parser, further parsers can be use to experiment with trickier grammars for heuristic exploration.

For example, given the famous [Dangling-Else](https://en.wikipedia.org/wiki/Dangling_else) grammar which

- is ambiguous and
- needs [left-factoring][LF] to be [LL(k)][LL] where `k < 4`,

<!-- Thanks to the powerful [GLL] algorithm, there is no need for **full backtracking**, which is a serious headache when designing performant and practical [LL(1)][LL] grammars. Even the highly notable [Parsec] in Haskell [cannot handle this with ease](http://hackage.haskell.org/package/parsec-3.1.11/docs/Text-Parsec-Prim.html#v:try).
-->

we declare a powerful *non-deterministic* [GLL parser][Gll] to process it directly:
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
    def stmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):    # differ stmt from stmt in param list
        return ('ite', stmt_1, stmt_2)
```
and it yields multiple legal results properly:

``` python
>>> P_IfThenElse.interpret('if 1 then if 2 then if 3 then x else yy else zzz')
[('ite', ('ite', ('it', 'x'), 'yy'), 'zzz'),
 ('ite', ('it', ('ite', 'x', 'yy')), 'zzz'),
 ('it', ('ite', ('ite', 'x', 'yy'), 'zzz'))]
```

## Parse Trees

In case only parse trees are needed, method bodies can be left emtpy. For exmaple, given the tricky grammar
```
A → A B
A → 
B → b B
B →
```

``` python
class A(metaclass=Earley.meta):
    b = r'b'
    def A(A, B): return
    def A(): return
    def B(b, B): return
    def B(): return
```


## Limitations

Though this module supplies many advantageous features, there are also limitations:

- Only legal Python identifier, rather than non-alphabetic symbols (like `<fo#o>`, `==`, `raise`, etc) can be used as symbols in grammar (seems no serious).

- Algorithms in pure Python lowers performance, but lots can be optimized. 


## Python 2 compatibility

The following version of the grammar in 1st section works for both Python 2 and Python 3, relying on decorators `cfg2` and `rule`:

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

The problem is that `type.__prepare__` method is not supported in Python 2, so that memoization of repeatedly declared method is not possible without logging work by the `rule` decorator. 

The resulted grammar instance is created by `cfg2` decorator which utilizes information logged by `rule`.

[Parsing]: https://en.wikipedia.org/wiki/Parsing "Parsing"
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
