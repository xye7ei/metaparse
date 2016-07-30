import preamble

from metaparse import *

table = {}

# Clean style
class G_Calc(metaclass=cfg):

    IGNORED = r'\s+'

    EQ  = r'='
    NUM = r'[0-9]+'
    ID  = r'[_a-zA-Z]\w*'
    POW = r'\*\*', 3
    MUL = r'\*'  , 2
    ADD = r'\+'  , 1

    # ERROR handler?

    def assign(ID, EQ, expr):
        table[ID] = expr

    def expr(NUM):
        return int(NUM)

    def expr(ID):
        return table[ID]

    def expr(expr_1, ADD, expr_2):
        return expr_1 + expr_2

    def expr(expr, MUL, expr_1):
        return expr * expr_1

    def expr(expr, POW, expr_1):
        return expr ** expr_1


# Handler style
class G_Calc():

    def IGNORED(lex: r'\v'):
        pass
    def IGNORED(lex: r'\\'):
        pass

    def ERROR(lex: r'\t'):
        print('ERROR')

    def UNRECOGNIZED(lex: r'.'):
        pass

    # Terminals
    def NUM(lex: r'\d+'):
        return int(lex)

    def ID(lex: r'[_a-zA-Z]\w*'):
        return lex

    def L(lex: r'\('):
        return lex
    def R(lex: r'\)'):
        return lex

    L2 = r'\['
    R2 = r'\]'

    def PLUS(lex: r'\+') -> 1:
        return lex
    def POW(lex: r'\*\*') -> 3:
        return lex
    def TIMES(lex: r'\*') -> 2:
        return lex
    
    # Nonterminals
    def assign(ID: r'[_a-zA-Z]\w*',
               EQ: '=',
               expr):
        table[ID] = expr

    def expr(NUM):
        return NUM

    def expr(expr_1, ADD: r'\+', expr_2):
        return expr_1 + expr_2


# Decorator style
def lex(pat, p=0):
    def _(func):
        return (func.__name__, pat, p, func)
    return _
        
class G_Calc():

    @lex(r'\s+', 3)
    def IGNORED(val):
        pass

    @lex(r'\t', 2)
    def ERROR(val):
        print('ERROR!')


# print(G_Calc.IGNORED)
# print(G_Calc.ERROR)


# With/If style
class g_Calc():

    def stmt():
        if (ID, '=', expr):
            table[ID] = expr
        elif (expr):
            pass

    def expr():
        with expr_1 * '+' * expr_2: # `withitems` is a list of three names
            x = expr_1 + expr_2
            return x * 1
        with expr_1, '*', expr_2: # `withitems` is a singleton list of one tuple
            return expr_1 * expr_2
        with NUM:
            return NUM


import inspect, ast, textwrap
stmt = ast.parse(textwrap.dedent(inspect.getsource(g_Calc.stmt))).body[0]

# i1,  = stmt.body
# i1.__dict__
# i1.orelse

# i1, i2 = stmt.body
# i1.__dict__
# i1.test


expr = ast.parse(textwrap.dedent(inspect.getsource(g_Calc.expr))).body[0]

# w1, w2, w3 = expr.body
# w1.__dict__
# w1.items
# w1.items[1].__dict__
# w1.items[1].context_expr.__dict__
# w1.body
# a1, r1 = w1.body
# a1.__dict__
# r1.__dict__

# w2.items

def with_to_func(lhs, w):
    args = []
    for j, item in enumerate(w.items):
        e = item.context_expr
        if isinstance(e, ast.Name):
            args.append(ast.Name(id=e.id, ctx=ast.Param()))
        elif isinstance(e, ast.Str):
            args.append(ast.Name(id='_%d' % j, ctx=ast.Param()))

    return ast.FunctionDef(
        name=lhs,
        lineno=w.lineno,
        args=ast.arguments(args=args,
                           defaults=[],
                           kw_defaults=[],
                           vararg=None,
                           kwarg=None,
                           kwonlyargs={},
                           lineno=0,
                           col_offset=0,
                       ),
        body=w.body,
        decorator_list=[],
    )

# stmt._fields
# stmt._attributes
# stmt.name
# stmt.returns
# stmt.args
# stmt.args.args
# f1 = with_to_func('expr', w1)
# ast.fix_missing_locations(f1)
# md = ast.Module([f1])
# # co = compile(md, '<ast>', 'exec')
