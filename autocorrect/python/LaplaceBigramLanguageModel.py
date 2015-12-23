from collections import Counter
from math import log

class LaplaceBigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.bigrams = Counter()
    self.unigrams = Counter()
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
        sentence_len = len(sentence.data)
        for prev_datum, next_datum in zip(sentence.data[:(sentence_len - 1)], sentence.data[1:sentence_len]):
            self.bigrams[(prev_datum.word, next_datum.word)] += 1

        for datum in sentence.data:
            self.unigrams[datum.word] += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    sentence_len = len(sentence)

    for prev_token, next_token in zip(sentence[:(sentence_len - 1)], sentence[1:sentence_len]):
        log_smoothed_unigram_counts = log(len(self.bigrams) + self.unigrams[prev_token])
        score += log(self.bigrams[(prev_token, next_token)] + 1) - log_smoothed_unigram_counts

    return score
