import sys
import os

# Include parent path for testing
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            os.pardir
        )))

class Hint(object):
    def __init__(self, info):
        self.info = info
    def __enter__(self):
        print('+++++++++++{}++++++++++++++'.format(self.info))
    def __exit__(self):
        print('-----------{}--------------'.format(self.info))
