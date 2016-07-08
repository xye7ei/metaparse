from metaparse import *


class _cfg2(object):

    _rule_list = []

    def flush():
        _cfg2._rule_list = []

    def rule(method):
        _cfg2._rule_list.append(Rule(method))

    def cfg2(cls_grammar):
        """Prepare alternative parser front-end functionalities for Python 2
        environment without metaprogramming tricks.

        In order to ease the use, a shared instance of rule_list is
        referred in this method. Each time after the decorator @cfg2
        is called and ended, this list is flushed. After that the next
        call of @rule would log Rule instance in the fresh list.

        """

        # In Python 2, OrderedDict is not easy to use.
        # lexes = OrderedDict()
        lexes = []
        lexpats = []
        rules = []
        attrs = []

        for k, v in cls_grammar.__dict__.items():
            # Ignore Built-ins
            if k.startswith('__') and k.endswith('__'):
                continue
            # Lexical declaration.
            if isinstance(v, str) and not k.startswith('_'):
                if k in lexes:
                    raise GrammarError('Repeated declaration of lexical symbol {}.'.format(k))
                lexes.append(k)
                lexpats.append(v)
            # Attributes
            elif not isinstance(v, Rule) and k.startswith('_'):
                attrs.append((k, v))

        for rule in _cfg2._rule_list:
            if rule not in rules:
                rules.append(rule)
            else:
                _cfg2.flush()
                raise GrammarError('Repeated declaration of Rule {}.'.format(rule))
        _cfg2.flush()

        # Default matching order of special patterns:

        # Always match IGNORED secondly after END, if it is not specified;
        if IGNORED not in lexes:
            # lexes.move_to_end(IGNORED, last=False)
            lexes.append(IGNORED)
            lexpats.append(IGNORED_PATTERN_DEFAULT)

        # Always match END first
        # END pattern is not overridable.
        # lexes[END] = END_PATTERN_DEFAULT
        lexes.insert(0, END)
        lexpats.insert(0, END_PATTERN_DEFAULT)
        # lexes.move_to_end(END, last=False)

        # Always match ERROR at last
        # It may be overriden by the user.
        if ERROR not in lexes:
            # lexes[ERROR] = ERROR_PATTERN_DEFAULT
            lexes.append(ERROR)
            lexpats.append(ERROR_PATTERN_DEFAULT)

        return Grammar(OrderedDict(zip(lexes, lexpats)), rules, attrs)


rule = _cfg2.rule
cfg2 = _cfg2.cfg2


@cfg2
class S:

    L1 = r'\('
    R1 = r'\)'
    SYMBOL = r'[^\(\)\s]+'

    @rule
    def sexp(SYMBOL):
        return SYMBOL

    @rule
    def sexp(L1, slist, R1):
        return slist

    @rule
    def slist():
        return ()

    @rule
    def slist(sexp, slist):
        return (sexp,) + slist



psr = GLR(S)
psr = LALR(S)

inp = """
(  (a b) c (d )  e )
"""

res = psr.interpret(inp)

pp.pprint(res)

