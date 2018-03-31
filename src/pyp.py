#!/usr/bin/python3
# Joan Gines i Ametlle

import numpy as np
import numpy.random as npr

# This file implements:
#   PYPrior: A class that represents the prior distribution over parameters in a PYP
#   PYP: A class that represents the CRP view of a Pitman-Yor Process

# Notes:
#   Operations for sampling are implemented as described in Teh (2006):
#     * Appendix B: Sampling for seating arrangements (inference)
#     * Appendix C: Sampling for parameters
#   The notation used in this script is also the same (as much as possible)

class PYPrior(object):

    def __init__(self, discount, strength):
        # the discount parameter has a prior discount ~ Beta(a, b)
        self.a = 1.0
        self.b = 1.0
        self.discount = discount
        # the strength parameter has a prior strength ~ Gamma(alpha, beta)
        self.alpha = 1.0
        self.beta = 1.0
        self.strength = strength

    # assign a new value to model parameters at the current level
    def sample_hyperparameters(self):
        # obtain the discount parameter by sampling from a Beta distribution
        self.discount = npr.beta(self.a, self.b)
        # ensure discount is in (0, 1)
        if self.discount <= 0 or self.discount >= 1:
            self.discount = 0.8
        # obtain the strength parameter by sampling from a Gamma distribution
        self.strength = npr.gamma(self.alpha, self.beta)
        # ensure strength is in (0, Inf) and normalize
        if self.strength <= 0:
            self.strength = 10.0
        self.strength = np.log10(self.strength)


class PYP(object):

    def __init__(self, base, prior):
        # base distribution for the Pitman-Yor Process
        self.base = base
        # prior distribution over parameters
        self.prior = prior
        # for each dish w, save the number of customers for each table k serving w
        self.dish2counts = {}
        # for each dish w, save the number of customers eating w at any table
        self.dish2customers = {}
        # number of tables in the current restaurant
        self.num_tables = 0
        # number of customers in the current restaurant
        self.num_customers = 0

    # add a new customer eating dish w at the table with index k
    def _add_customer(self, w, k, open_table=False):
        # add a new dish to the restaurant if needed
        if w not in self.dish2counts:
            self.dish2counts[w] = []
            self.dish2customers[w] = 0
        # increase table counts
        if open_table:
            # add a new table to the restaurant
            self.dish2counts[w] += [1]
            self.num_tables += 1
            # add customer to the previous level process
            self.base.increment(w)
        else:
            # sit the customer in an existing table
            self.dish2counts[w][k] += 1
        # increase customer counts
        self.dish2customers[w] += 1
        self.num_customers += 1

    # remove a customer eating dish w at the table with index k
    def _remove_customer(self, w, k):
        # discount customer
        self.dish2counts[w][k] -= 1
        self.dish2customers[w] -= 1
        self.num_customers -= 1
        # remove table k if no customers are left
        if not self.dish2counts[w][k]:
            del self.dish2counts[w][k]
            self.num_tables -= 1
            # remove customer from the previous level process
            self.base.decrement(w)
        # remove dish w if it is not served at any table
        if not len(self.dish2counts[w]):
            del self.dish2counts[w]
            del self.dish2customers[w]

    # add a new customer eating dish w into the restaurant
    def increment(self, w):
        # determine to which table dish w needs to be assigned
        open_table = not w in self.dish2counts
        table_index = 0
        if not open_table:
            # share a table with probability proportional to p_share
            p_share = self.dish2customers[w] - self.prior.discount * len(self.dish2counts[w])
            # open a new table with probability proportional to p_new
            p_new = (self.prior.strength + self.prior.discount * self.num_tables) * self.base.word_prob(w)
            # decide on the result with a cumulative probability approach
            r = npr.uniform(0, p_share + p_new)
            if r < p_new:
                open_table = True
            else:
                acc = p_new
                for cuwk in self.dish2counts[w]:
                    if r < acc + cuwk - self.prior.discount:
                        break
                    if table_index == len(self.dish2counts[w]) - 1:
                        break
                    acc += cuwk - self.prior.discount
                    table_index += 1
        # add new customer
        self._add_customer(w, table_index, open_table)

    # remove a customer eating dish w from the restaurant
    def decrement(self, w):
        # determine from which table we need to remove a customer
        r = npr.randint(0, self.dish2customers[w])
        table_index = 0
        # use cumulative probability to decide
        for cuwk in self.dish2counts[w]:
            if r < cuwk:
                break
            r = r - cuwk
            table_index += 1
        # remove customer
        self._remove_customer(w, table_index)

    # return the probability that the next word after the current context will be w
    def word_prob(self, w):
        p = 0.0
        if w in self.dish2counts:
            p = self.dish2customers[w] - self.prior.discount * len(self.dish2counts[w])
        p += (self.prior.strength + self.prior.discount * self.num_tables) * self.base.word_prob(w)
        return p / (self.prior.strength + self.num_customers)

    # update the auxiliary variables and accumulate their values
    def update_variables(self):
        if self.num_tables >= 2:
            # auxiliary variables of type xu
            xu = npr.beta(self.prior.strength + 1, self.num_customers - 1)
            self.prior.beta -= np.log10(xu)
            # auxiliary variables of type yui
            for i in range(1, self.num_tables):
                yui = npr.binomial(1, self.prior.strength / (self.prior.strength + self.prior.discount * i))
                self.prior.a += (1 - yui)
                self.prior.alpha += yui
        # auxiliary variables of type zuwkj
        for w in self.dish2counts:
            for cuwk in self.dish2counts[w]:
                if cuwk >= 2:
                    for j in range(1, cuwk):
                        zuwkj = npr.binomial(1, (j - 1) / (j - self.prior.discount))
                        self.prior.b += (1 - zuwkj)

