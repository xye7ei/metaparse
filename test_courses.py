from earley import earley

def fappend(l, x):
    l.append(x)
    return l

class Gcourses(metaclass=earley):

    """
    Grammar to assign multiple numbers to precedend course name.
    Examples:

"CS 2110"                        => ("CS", 2110) # 0

"CS 2110 and INFO 3300"          => [("CS", 2110), ("INFO", 3300)] # 1
"CS 2110, INFO 3300"             => [("CS", 2110), ("INFO", 3300)] # 1
"CS 2110, 3300, 3140"            => [("CS", 2110), ("CS", 3300), ("CS", 3140)] # 1

"CS 2110 or INFO 3300"           => [[("CS", 2110)], [("INFO", 3300)]] # 2

"MATH 2210, 2230, 2310, or 2940" => [[("MATH", 2210), ("MATH", 2230), ("MATH", 2310)], [("MATH", 2940)]] # 3

    """

    # IGNORED = r'[ \t]+|(,)|(and)'
    NAME    = r'[A-Z]+'
    NUMBER  = r'\d{4}'
    OR      = r'or'


    # info -> headed
    def info(headed):                return headed

    # info -> conj
    def info(conj):                  return conj

    # info -> disj
    def info(disj):                  return disj

    # headed -> NAME nums
    def headed(NAME, nums):          return [(NAME, x) for x in nums]

    # nums -> nums NUMBER
    def nums(nums, NUMBER):          return fappend(nums , NUMBER)
    # def nums(nums, NUMBER):          return nums + [NUMBER]

    # nums -> NUMBER
    def nums(NUMBER):                return [NUMBER]

    # conj -> headed headed
    def conj(headed1, headed2):      return headed1 + headed2

    # disj -> headed OR headed
    def disj(headed1, OR, headed2):  return [headed1, headed2]

    # disj -> headed OR nums
    def disj(headed1, OR, nums):     return [headed1, [(headed1[0][0], n) for n in nums]]

import pprint as pp

gcrs = Gcourses

inp = "CS 2110"
# import pdb
# pdb.set_trace()
inp = "CS 2110 and INFO 3300"
gcrs.eval(inp)
inp = "CS 2110, INFO 3300"
gcrs.eval(inp)
inp = "CS 2110, 3300, 3140"
gcrs.eval(inp)
inp = "CS 2110 or INFO 3300"
gcrs.eval(inp)
gcrs.parse(inp)

# Compare forms with same semantics...

inp = "MATH 2210, 2230, 2310 or 2940"
print(repr(inp))
s1 =  Gcourses.parse(inp)
v1 =  Gcourses.eval(inp)
pp.pprint(v1)

inp = "MATH 2210, 2230, 2310, or 2940"
print(repr(inp))
s2 = Gcourses.parse(inp)
v2 =  Gcourses.eval(inp)
pp.pprint(v2)

# pp.pprint(v2, depth=2)

assert s1 == s2
assert v1 == v2

