#!/usr/bin/python3
# Joan Gines i Ametlle

import pickle
import argparse
import numpy as np
from corpus import Corpus

# This file implements:
#   The evaluation procedure for a Hierarchical Pitman-Yor Language Model

# Notes:
#   This script requires a trained model file (please refer to train.py)
#   It is assumed that the test data contains one sentence per line

def parse_arguments():
    # messages to display in the help screen (--help)
    msg_main = 'Evaluate a Hierarchical Pitman-Yor Language Model (HPYLM)'
    msg_test = 'Test file (system path)'
    msg_model = 'Model file obtained with train.py (system path)'

    # get value for each argument
    parser = argparse.ArgumentParser(description = msg_main)
    parser.add_argument('--test', help = msg_test, required = True)
    parser.add_argument('--model', help = msg_model, required = True)
    return parser.parse_args()


def main():
    # parse input arguments
    args = parse_arguments()
    test_file = args.test
    model_file = args.model

    # load trained model
    with open(model_file, 'rb') as mfile:
        model = pickle.load(mfile)

    # read the test file and build test data
    corpus = Corpus(test_file, model.order)

    # report log-likelihood and perplexity on the test data
    logprob = 0
    for ngram in corpus.ngrams:
        # compute and add the log-probability of each word given the context
        logprob += np.log10(model.word_prob(ngram[:-1], ngram[-1]))

	# compute perplexity
	# the vocabulary is closed (0 OOVs), so this is equivalent to SRILM's ppl
    ppl = np.power(10, -logprob / (corpus.num_words + corpus.num_sents))

    # get all prior parameter values (for reference only)
    hyperparams = model.get_hyperparameters()

    # show evaluation results and other information
    print('Number of words:       \t' + str(corpus.num_words))
    print('Number of sentences:   \t' + str(corpus.num_sents))
    for i in range(model.order):
        d = '{0:.8f}'.format(hyperparams[i][0])
        theta = '{0:.8f}'.format(hyperparams[i][1])
        print('G{'+ str(i+1) +'} prior (d, theta):\t' + d + ', ' + theta)
    print('Log-probability:       \t' + str(logprob))
    print('Perplexity:            \t' + str(ppl))


if __name__ == '__main__':
    main()

