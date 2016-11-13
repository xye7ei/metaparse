from eg_demo import *

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


