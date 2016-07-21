from subprocess import call
from glob import glob

fs = glob('./test_*.py')

# print(fs)
for f in fs:
    call(['python', f])

cmd = 'for %%f in ("test_*.py") do python "%%~nxf";'

