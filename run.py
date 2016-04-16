"""
Usage:
    run.py perceptron --train=<train> --test=<test>

Options:
    --train Path to training data, txt file.
    --test Path to test data, txt file.
"""

from redes_neurais.resources.manager import run_perceptron
import docopt
import logging


logger = logging.getLogger('neural_network')


def run():
    try:
        args = docopt.docopt(__doc__)
        if args['perceptron']:
            run_perceptron(args['--train'], args['--test'])
    except docopt.DocoptExit as e:
        print e.message

if __name__ == "__main__":
    run()
