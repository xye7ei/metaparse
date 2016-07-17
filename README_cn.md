一个（可能是世界上最易用的Parser），欢迎来测


如果你会Python，并接触过Parser和Interpreter的知识，知道上下文无关语言（[CFG][]），那你很可能对这个模块感兴趣。

@陈天 老师写过一篇文推广Parser的概念，颇受欢迎。传送门：https://zhuanlan.zhihu.com/p/20178871

就像文中提到的，有没有想象过像使用正则一般地使用CFG Parser？在Python中，事情可以很simply but elegantly地解决。

<!-- （什么？你说性能？啥？对不起刚才信号不太好，以后再说吧我们先继续~）
 -->

话不多说，我们来写带变量的~~残缺的~~计算器，支持加、乘、乘方和赋值：


# 示例：计算器

第一步，设计文法：
```
stmt → ID = expr
stmt → expr
expr → NUM
expr → ( expr )
expr → expr₁ + expr₂
expr → expr₁ * expr₂
expr → expr₁ ** expr₂
```

第二步，用Python方法签名形式模拟该文法...
``` python
def stmt(ID, EQ, expr): ...
def stmt(expr): ...
def expr(ID): ...
def expr(NUM): ...
def expr(L, expr, R): ...
def expr(expr_1, ADD, expr_2): ...
def expr(expr_1, MUL, expr_2): ...
def expr(expr_1, POW, expr_2): ...
```

会不会给人一种天然统的感觉...？

第三步，基于`metaparse`做三个微小的工作：词法声明，句法声明及实现文法语义：
``` python
# 导入cfg元类和LALR解析器
from metaparse import cfg, LALR

# 全局变量
table = {}

class Calc(metaclass=cfg):

    # 词法（基于正则，顺序匹配）
    IGNORED = r'\s+'

    L   = r'\('
    R   = r'\)'
    EQ  = r'='
    NUM = r'\d+'
    ID  = r'\w+'
    POW = r'\*\*', 3    # 指配Token的优先级，帮助LALR消歧
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    # 句法和语义（句法制导翻译，即Yacc的SDT风格）

    def stmt(ID, EQ, expr):
        table[ID] = expr
        return expr

    def stmt(expr):
        return expr

    def expr(ID):
        if ID in table:
            return table[ID]
        else:
            raise ValueError('ID yet bound: {}'.format(ID))

    def expr(NUM):
        return int(NUM)

    def expr(L, expr, R):
        return expr

    def expr(expr_1, ADD, expr_2):   # TeX风格下标区分重名的形参
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):     # 下标足以区分即可
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1
```

第四步，用这个Grammar构建Parser：

```
>>> type(G_Calc)
<class 'metaparse.Grammar'>
>>> Calc = LALR(G_Calc)
```

然后变量计算器就完工了，在REPL中亦可赛艇：

``` python
>>> calc.interpret(' (3) ')
3
>>> calc.interpret(' x = 3 ')
3
>>> calc.interpret(' y = 4 * x ** (2 + 1) * 2')
216
>>> table
{'x': 3, 'y': 216}
```

# 内核：元编程

由于对元类`metaclass`的支持问题，以上写法只在Python 3里面有效。元类的目的很简单：

- 在声明类的过程中，按顺序记录一个类的属性；
- 允许记录重名属性实例。

用来记录这些的结构可以是一个key-value-pair list（即assoc-list），支持`__setitem__`和`__getitem__`即可。详情可见官方文档中`__prepare__`和`__new__`的说明，以及David Beazley的[教程](http://www.dabeaz.com/py3meta/)。

成功获取属性列表之后，借助`inspect.getargspec(func)`得到其中函数对象的签名，利用它们构造一个兼具形式和语义的`Rule`对象，基于它们来构造`Grammar`对象。


## 深度元编程和`ast`模块

若不依赖`metaclass`（比如在Python 2中），我们放弃`class`结构，转而使用`def`结构。`metaparse`提供了一个decorator，如下使用可以得到和以上相同的结果：

``` python
@cfg.v2
def Calc():

    IGNORED = r'\s+'
    ...
    <same-stuff>
    ...

```

简而言之`cfg.v2`做了几件事：

- 借助`inspect.getsource`得到源码字符串；
- 用`ast.parse`得到编译的语法树；
- 遍历语法树， 搜集信息和对象来构造`Grammar`。

关于`ast`的文档比较少，但是它已经被广泛用于构造DSL和模板引擎之类的工具。篇幅所限，就不再展开。


# 关于Parser后端

上面提到的这些元编程的tricks只提供了一种便捷地方式来声明CFG，而后端的Parsing算法可以有多种实现。目前模块实现的除了LALR，还有简单孱弱的Weak-LL(1)（之所以weak因为没有用*FOLLOW-SET*）以及可以处理歧义文法的GLR/Earley。

对于某种LL(1)文法，我们可以使用`WLL1`，比如可爱的*LISP*:
``` python
class LISP(metaclass=cfg):

    IGNORED = r'\s+'
    LAMBDA = r'\(\s*lambda'
    LEFT   = r'\('
    RIGHT  = r'\)'
    SYMBOL = r'[^\(\)\s]+'

    def sexp(var):
        return var
    def sexp(abst):
        return abst
    def sexp(appl):
        return appl

    def var(SYMBOL):
        return SYMBOL
    def abst(LAMBDA, LEFT, parlist, RIGHT_1, sexp, RIGHT_2):
        return ('LAMBDA', parlist, sexp)
    def appl(LEFT, sexp, sexps, RIGHT):
        return [sexp, sexps]

    def parlist(SYMBOL, parlist):
        return [SYMBOL] + parlist
    def parlist():
        return []

    def sexps(sexp, sexps):
        return [sexp] + sexps
    def sexps():
        return []

p_lisp = WLL1(LISP)

>>> p_lisp.interpret('(lambda (x y) (+ x y))')
('LAMBDA', ['x', 'y'], ['+', ['x', 'y']])
```

或者对于歧义文法用GLR，比如[Dangling-Else][]:
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

p_ite = GLR(G_IfThenElse)

>>> p_ite.interpret_many('if 1 then if 2 then if 3 then a else b else c')
[('i-t', ('i-t-e', ('i-t-e', 'a', 'b'), 'c')),
 ('i-t-e', ('i-t', ('i-t-e', 'a', 'b')), 'c'),
 ('i-t-e', ('i-t-e', ('i-t', 'a'), 'b'), 'c')]
```

上面计算器的文法也是歧义文法，只不过借助了Token的优先级指配消除了歧义，具体概念不再赘述。


# 后话

这个模块目前还不太成熟，但我个人认为前端很好用，而后端的算法也可以在未来不断优化。希望有兴趣的童鞋可以帮助测试这个模块并给予反馈。`Github`我还是个菜鸡，大家可以去开ISSUE，互相交流学习。

[项目地址](https://github.com/Shellay/metaparse/)，欢迎fork。


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
