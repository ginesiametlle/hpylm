#!/usr/bin/python3
# Joan Gines i Ametlle

from nltk.corpus import cmudict
from pyp import PYPrior, PYP

# This file implements:
#   G0Uniform: A class that represents a uniform distribution over words
#   GuBase: A class that represents a base for a distribution over distributions
#   HPYLM: A class that represents a Hierarchical Pitman-Yor Language Model

# Notes:
#   The global base distribution G{0} is the uniform distribution
#   HPYLM is built recursively, and G{u} is drawn as a PYP from G{u-1}

class G0Uniform(object):

    def __init__(self, V):
        # size of the vocabulary
        self.V = V

    # add a word w (nothing to do for G0)
    def increment(self, w):
        pass

    # remove a word w (nothing to do for G0)
    def decrement(self, w):
        pass

    # return the probability that the next word will be w
    def word_prob(self, w):
        # each word is equally probably under G0
        return 1. / self.V

    # sample parameters (nothing to do for G0)
    def sample_hyperparameters(self):
        pass


class GuBase(object):

    def __init__(self, context, backoff):
        # reduced context in G{u}
        self.context = context
        # back-off distribution for G{u}, that is G{u-1}
        self.backoff = backoff

    # add a word w
    def increment(self, w):
        return self.backoff.increment(self.context, w)

    # remove a word w
    def decrement(self, w):
        return self.backoff.decrement(self.context, w)

    # return the probability that the next word after the context will be w
    def word_prob(self, w):
        return self.backoff.word_prob(self.context, w)


class HPYLM(object):

    def __init__(self, order):
        # order of the model
        self.order = order
        # prior distribution over parameters (initially discount=0.8, strength=1.0)
        self.prior = PYPrior(0.8, 1.0)
        # back-off distribution
        if order == 1:
            self.backoff = G0Uniform(len(cmudict.words()) + 2)
        else:
            self.backoff = HPYLM(order-1)
        # mapping of contexts to the corresponding Pitman-Yor Process
        self.u2pyp = {}

    # add a word w after context u (recursively)
    def increment(self, u, w):
        # create a new PYP using the back-off distribution as base if needed
        if u not in self.u2pyp:
            if self.order == 1:
                self.u2pyp[u] = PYP(self.backoff, self.prior)
            else:
                self.u2pyp[u] = PYP(GuBase(u[1:], self.backoff), self.prior)
        self.u2pyp[u].increment(w)

    # remove a word w after context u (recursively)
    def decrement(self, u, w):
        if u in self.u2pyp:
            self.u2pyp[u].decrement(w)

    # return the probability that the next word after u will be w
    def word_prob(self, u, w):
        # resort to a new PYP using the back-off distribution as base if needed
        if u not in self.u2pyp:
            if self.order == 1:
                return PYP(self.backoff, self.prior).word_prob(w)
            else:
                return PYP(GuBase(u[1:], self.backoff), self.prior).word_prob(w)
        return self.u2pyp[u].word_prob(w)

    # sample the model parameters
    def sample_hyperparameters(self):
        # compute the values for the auxiliary variables in G{u}
        for u in self.u2pyp:
            self.u2pyp[u].update_variables()
        # assign new values to parameters in G{u}
        self.prior.sample_hyperparameters()
        # continue sampling for the back-off distribution G{u-1}
        self.backoff.sample_hyperparameters()

    # return all model parameters as a list of pairs (for reference only)
    def get_hyperparameters(self):
        if self.order == 1:
            return [(self.prior.discount, self.prior.strength)]
        else:
            return self.backoff.get_hyperparameters() + [(self.prior.discount, self.prior.strength)]

