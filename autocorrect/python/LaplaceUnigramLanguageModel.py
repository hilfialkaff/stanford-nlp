from collections import Counter
from math import log

class LaplaceUnigramLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.words = Counter()
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
        for token in sentence.data:
            self.words[token.word] += 1
    self.total = len(self.words)

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    log_smoothed_words = log(sum(self.words.values()) + self.total)

    for token in sentence:
        score += log(self.words[token] + 1) / log_smoothed_words
    return score
