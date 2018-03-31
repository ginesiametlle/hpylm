#!/usr/bin/python3
# Joan Gines i Ametlle

import string
from nltk.util import ngrams
from nltk.corpus import cmudict

# This file implements:
#   Corpus: A class that wraps a data file and provides useful methods

class Corpus(object):

    def __init__(self, ifile, order):
        # input file
        self.ifile = ifile
        # number of sentences and words in the corpus
        self.num_sents = 0
        self.num_words = 0
        # start, stop and unknown-word special symbols
        self.start_sym = '<s>'
        self.stop_sym = '</s>'
        self.unk_sym = '<unk>'
        # mapping from words to integers and contrariwise
        self.word2id = {self.start_sym: 0, self.stop_sym: 1, self.unk_sym: 2}
        self.id2word = [self.start_sym, self.stop_sym, self.unk_sym]
        self._build_vocabulary()
        # n-grams extracted from the input data
        self.order = order
        self.ngrams = []
        self._build_ngrams(open(ifile).readlines())

    # provide an iterator over encoded sentences in the corpus
    def __iter__(self):
        return iter(self._encode_sent(s) for s in open(self.ifile).readlines())

    # return whether a word w appears in the corpus
    def contains(self, w):
        return w in self.word2id

    # build a vocabulary based on the CMU dictionary
    def _build_vocabulary(self):
        for word in cmudict.words():
            if not self.contains(word):
                self.word2id[word] = len(self.id2word)
                self.id2word.append(word)

    # return the encoding of a word w
    def _encode_word(self, w):
        if self.contains(w.lower()):
            return self.word2id[w.lower()]
        # words not in the vocabulary are mapped to the <unk> token
        return self.word2id[self.unk_sym]

    # return the encoding of a sentence s
    def _encode_sent(self, s):
        words = [w for w in s.split()]
        return [self._encode_word(w) for w in words]

    # construct n-grams from a stream
    def _build_ngrams(self, stream):
        # split input into sentences
        for sent in stream:
            self.num_sents += 1
            # split each sentence into words
            words = [w for w in sent.split()]
            self.num_words += len(words)
            # compute n-grams based on word tokens
            for gram in ngrams(words, self.order,
                               pad_left = True, pad_right = True,
                               left_pad_symbol = self.start_sym, right_pad_symbol = self.stop_sym):
                # limit right-padding for n-grams of order larger than 2
                if self.order > 2:
                    if gram[-1] == self.stop_sym and gram[-2] == self.stop_sym:
                        continue
                self.ngrams.append(tuple([self._encode_word(w) for w in gram]))

