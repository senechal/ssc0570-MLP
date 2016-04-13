"""
Usage:
    run.py start
"""

from mlp.mlp import start
import docopt
import logging


logger = logging.getLogger('mlp')


def run():
    try:
        args = docopt.docopt(__doc__)
        if args['start']:
            start()
    except docopt.DocoptExit as e:
        print e.message

if __name__ == "__main__":
    run()
