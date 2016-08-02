import preamble
from metaparse import cfg, LALR, Symbol
from collections import OrderedDict
from pprint import pprint

class Helper:

    terms = OrderedDict()
    _c = -1

    def reset():
        Helper._c = -1
        Helper.terms = OrderedDict()

    def get_term(lit):
        Helper._c += 1
        if lit not in Helper.terms:
            term = Symbol('TM{}'.format(Helper._c))
            Helper.terms[lit] = term
        return Helper.terms[lit]


class YACC(metaclass=cfg):

    IGNORED = r'\s+'
    IGNORED = r'\/\*[^(\*/)]*\*\/'
    # IGNORED = r'\{[^\}]*\}'
    
    ALT = r'\|'
    DRV = r':'
    SEMI = r';'
    
    BODY = r'\{[^\}]*\}'

    ID = r'[_a-zA-Z]\w*'
    TERM1 = r"\'[^\']*\'"
    TERM2 = r'\"[^\"]*\"'

    def grammar(rules):
        terms = ['    {} = r{}'.format(tok, repr(pat[1:-1])) for pat, tok in Helper.terms.items()]
        gen = '\n'.join([
            'from metaparse import LALR',
            '',
            'class G(metaclass=LALR.meta):',
            '',
            *terms,
            '',
            *rules,
        ])
        return gen

    def rules(): return []
    def rules(rules, rule):
        return rules + rule
    
    def term(TERM1):
        return Helper.get_term(TERM1)
    def term(TERM2):
        return Helper.get_term(TERM2)

    def rule(ID, DRV, alts, SEMI):
        r_defs = []
        for seq, bdy in alts:
            r_def = '    def {}{}:\n        r"""{}"""'.format(
                ID,
                seq,
                repr(bdy),
            )
            r_defs.append(r_def)
        return r_defs

    def alts(alts, ALT, alt):
        alts.append(alt)
        return alts
    def alts(alt):
        return [alt]

    def alt(seq):
        return (seq, '')
    def alt(seq, BODY):
        return (seq, BODY)

    def seq(seq, symbol):
        return seq + (symbol,)
    def seq():
        return ()

    def symbol(term):
        return term
    def symbol(ID):
        return Symbol(ID)


eg = """
input:    /* empty */
        | input line
;

line:     '\n'
        | exp '\n'  { printf ("\t%.10g\n", $1); }
;

exp:      NUM             { $$ = $1;         }
        | exp exp '+'     { $$ = $1 + $2;    }
        | exp exp '-'     { $$ = $1 - $2;    }
        | exp exp '*'     { $$ = $1 * $2;    }
        | exp exp '/'     { $$ = $1 / $2;    }
      /* Exponentiation */
        | exp exp '^'     { $$ = pow ($1, $2); }
      /* Unary minus    */
        | exp '-'         { $$ = -$1;        }
;    
"""

# pprint([*YACC.tokenize(eg, True)])

yacc = LALR(YACC)
tr = yacc.parse(eg)
res = yacc.interpret(eg)

# pprint(yacc.grammar.lexers)
# pprint(yacc)
# pprint(tr)
print()
print(res)
