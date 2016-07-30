import preamble

import ast

import inspect
import textwrap

from metaparse import *

X = 100
# If style
class G:

    def stmt():
        if (ID, '=', expr):
            table[ID] = expr
        if expr:
            pass

    def expr():
        "Interleaving terminal literals should be logged into the tokenizer."
        if (expr_1, '+', expr_2): # `withitems` is a list of three names
            # with pre:
            #     print('Prepare to add!')
            # with post:
            #     x = expr_1 + expr_2
            #     return x + X
            x = expr_1 + expr_2
            return x + X
        if (expr_1, '*', expr_2): # `withitems` is a singleton list of one tuple
            return expr_1 * expr_2 + X
        if num:
            return num + X
        # if 0:
        #     return 0

    # def num():
    #     if r'\d+':
            

def if_2_funcdef(name, t_if, glb):

    args = []
    lineno = t_if.lineno
    col_offset = t_if.col_offset
    body = t_if.body

    symbs = []
    if isinstance(t_if.test, (ast.Tuple, ast.List)):
        elts = t_if.test.elts
    elif isinstance(t_if.test, (ast.Name, ast.Str)):
        elts = [t_if.test]
    else:
        raise ValueError('Invalid RHS.')

    for j, elt in enumerate(elts):
        if isinstance(elt, ast.Name):
            par_name = elt.id
        elif isinstance(elt, ast.Str):
            par_name = elt.s
        arg = ast.arg(
            arg=par_name,
            annotation=None,
            lineno=lineno,
            col_offset=col_offset
        )
        args.append(arg)
        symbs.append(par_name)

    arguments = ast.arguments(
        args = args,
        vararg = None,
        kwarg = None,
        kwonlyargs = [],
        kw_defaults = [],
        defaults = []
    )

    fd = ast.FunctionDef(
        name = name,
        args = arguments,
        body = body,
        lineno = lineno,
        col_offset = col_offset - 4,
        decorator_list = [],
    )
    fd = ast.fix_missing_locations(fd)

    ctx = {}
    co = compile(ast.Module([fd]), '<ast>', 'exec')
    exec(co, glb, ctx)

    return Rule(name, symbs, ctx[name])


def def_2_rules(f):

    n = f.__name__
    src = inspect.getsource(f)
    src = textwrap.dedent(src)
    fd = ast.parse(src).body[0]

    iis = []
    for ii in fd.body:
        if isinstance(ii, ast.If):
            iis.append(ii)
            while ii.orelse:
                ii = ii.orelse[0]
                iis.append(ii)

    rules = []
    for ii in iis:
        r = if_2_funcdef(n, ii, f.__globals__)
        rules.append(r)

    return rules

exprs = def_2_rules(G.expr)

from pprint import pprint
pprint(exprs)
e1, e2, e3 = exprs

print(e1.seman(3, None, 2))
print(e2.seman(3, None, 2))
print(e3.seman(9))
