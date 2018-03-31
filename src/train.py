#!/usr/bin/python3
# Joan Gines i Ametlle

import pickle
import argparse
from hpylm import HPYLM
from corpus import Corpus

# This file implements:
#   The training procedure for a Hierarchical Pitman-Yor Language Model

# Notes:
#   The model obtained is stored in a file after training is finished
#   It is assumed that the training data contains one sentence per line

def parse_arguments():
    # messages to display in the help screen (--help)
    msg_main = 'Train a Hierarchical Pitman-Yor Language Model (HPYLM)'
    msg_train = 'Training file (system path)'
    msg_out = 'Output model file (system path)'
    msg_order = 'Length of contexts (default: 3)'
    msg_niter = 'Number of iterations (default: 10)'

    # get value for each argument
    parser = argparse.ArgumentParser(description = msg_main)
    parser.add_argument('--train', help = msg_train, required = True)
    parser.add_argument('--out', help = msg_out, required = True)
    parser.add_argument('--order', type = int, help = msg_order)
    parser.add_argument('--niter', type = int, help = msg_niter)
    return parser.parse_args()


def main():
    # parse input arguments
    args = parse_arguments()
    train_file = args.train
    model_file = args.out
    order = 3
    if args.order:
        order = args.order
    num_iterations = 10
    if args.niter:
        num_iterations = args.niter

    # read the training file and build training data
    corpus = Corpus(train_file, order)

    # instantiate the initial model
    model = HPYLM(order)

    # train the model (run Gibbs sampling + parameter sampling)
    for it in range(num_iterations):
        print('Training iteration ', it + 1, '/', num_iterations)
        print('   Sampling for seating arrangements...')
        for ngram in corpus.ngrams:
            if it > 0:
                model.decrement(ngram[:-1], ngram[-1])
            model.increment(ngram[:-1], ngram[-1])
        print('   Sampling for model parameters...')
        model.sample_hyperparameters()

    # output the resulting model
    with open(model_file, 'wb') as mfile:
        pickle.dump(model, mfile, protocol = -1)


if __name__ == '__main__':
    main()

