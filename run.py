"""
Usage:
    run.py mlp --train=<train> --test=<test> --config=<config>

Options:
    --train Path to training data, txt file.
    --test Path to test data, txt file.
    --config Json  conofiguration for MLP.
"""

from redes_neurais.resources.manager import run_mlp
import docopt




def run():
    try:
        args = docopt.docopt(__doc__)
        if args["mlp"]:
            run_mlp(args['--config'], args['--train'], args['--test'])
    except docopt.DocoptExit as e:
        print e.message

if __name__ == "__main__":
    run()
