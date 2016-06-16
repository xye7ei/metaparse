import grammar

from collections import namedtuple
from collections import OrderedDict


class XMLGrammar(metaclass=grammar.cfg):

    IGNORED = r'\s'
    SLASH   = r'/'
    LEFT    = r'<'
    RIGHT   = r'>'
    EQ      = r'='
    TEXT    = r'[\w\+-]+'
    QUOTED  = r'"[^\"]*"'

    def tag(tag_open, tags, tag_close):
        return (tag_open[0], tag_open[1], tags)
    def tag(LEFT, TEXT, attrs, SLASH, RIGHT):
        return (TEXT, attrs, [])
    
    def tag_open(LEFT, TEXT, attrs, RIGHT):
        return (TEXT, attrs)
    def tag_close(LEFT, SLASH, TEXT, RIGHT):
        return None

    def tags(tags, tag):
        return tags + [tag]
    def tags():
        return []

    def attrs(attrs, attr):
        return {**attrs, **attr}
    def attrs():
        return {}
    
    def attr(TEXT, EQ, QUOTED):
        return {TEXT: QUOTED}


class ParseTree(object):
    def __init__(self, item, subs):
        self.item = item
        self.subs = subs
    def __iter__(self):
        yield item
        yield subs
    def eval(self):
        # Only recursive eval possible
        return self.item.eval(*self.subs)
    def eval(self):
        return


class Earley(grammar.Grammar):

    def __init__(self, lexes_rules):
        super(Earley, self).__init__(lexes_rules)
        self.start_item = self.make_item(r=0, pos=0)

    def __repr__(self):
        return '{}{}'.format('Earley', super(Earley, self).__repr__())

    def parse_process(G, inp: str) -> [[grammar.Item]]:
        # Each state is a tuple:
        # (<item>, <subtrees>, <predictor-index>)
        # How to associate semantic behaviors to subtrees????
        State = lambda itm, args, i: (itm, args, i)

        S = [[(G.make_item(r=0, pos=0), [], 0)]]
        for k, (at, lex, lexval) in enumerate(G.tokenize(inp)):
            # Closuring
            z = 0
            while z < len(S[k]):
                jtm, j_args, j = S[k][z]
                # Prediction: Nonkernels
                if not jtm.ended():
                    for i_r, rule in enumerate(G.rules):
                        if rule.lhs == jtm.active():
                            new = State(G.make_item(i_r, pos=0), [], k)
                            if new not in S[k]:
                                S[k].append(new)
                # Completion
                else: # jtm.ended()
                    for itm, i_args, i in S[j]:
                        if not itm.ended():
                            if itm.active() == jtm.target():
                                new = State(itm.shifted(),
                                            i_args + [(jtm.target(), j_args)],
                                            i) # completed subtree
                                if new not in S[k]:
                                    S[k].append(new)
                z += 1

            S.append([])
            # Scanning/proceed for next States
            for jtm, j_args, j in S[k]:
                # # Top rule
                # if jtm.ended() and jtm.r == 0 and lex == grammar.END:
                #     S[k+1].append(State(jtm, j_args, j))
                # Non-top rule
                if not jtm.ended() and jtm.active() == lex:
                    S[k+1].append(State(jtm.shifted(), j_args + [lexval], j)) # completed
            if not S[k+1]:
                raise ValueError('Choked by {} at {}.'.format(lexval, at))
            elif len(S[k+1]) > 1:
                # print('Local ambiguity: parallel States.')
                pass
        return S
    

# g_xml = Earley(XMLGrammar)

class GIfThenElse(metaclass=grammar.cfg):
    # IGNORED = r'\s'
    IF      = r'if\s*'
    THEN    = r'then\s*'
    ELSE    = r'else\s*'
    EXPR    = r'\(\s*e\s*\)\s*'
    SINGLE  = r's'
    
    def stmt(SINGLE):
        return SINGLE
    def stmt(IF, EXPR, THEN, stmt):
        return (1, 'expr1', stmt)
    def stmt(IF, EXPR, THEN, stmt_1, ELSE, stmt_2):
        return (2, 'expr2', stmt_1, stmt_2)

g_if = Earley(GIfThenElse)

r1 = g_if.parse_process('if ( e) then    s')
r2 = g_if.parse_process('if (  e ) then    s else  s')
r3 = g_if.parse_process('if (e ) then  if ( e ) then  s   else s')

from pprint import pprint

# pprint(r1)
# pprint(r2)
# pprint(list(g_if.tokenize('if (e ) then  if   e then  s   else s')))
pprint(r3)
