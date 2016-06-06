"""
Usage:
    run.py mlp --train=<train> --test=<test> --config=<config>
    run.py som --train=<train> --test=<test> --config=<config>

Options:
    --train Path to training data, txt file.
    --test Path to test data, txt file.
    --config Json  configuration for the network.
"""

from redes_neurais.resources.manager import run_mlp, run_som
import docopt

def run():
    try:
        args = docopt.docopt(__doc__)
        if args["mlp"]:
            run_mlp(args['--config'], args['--train'], args['--test'])
        if args["som"]:
            run_som(args['--config'], args['--train'], args['--test'])
    except docopt.DocoptExit as e:
        print e.message

if __name__ == "__main__":
    run()
