from collections import Counter
from math import log

class GoodTuringSmoothingModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.word_counts = Counter() # How many times a word is seen
    self.inverted_count_index = Counter() # Mapping between n -> number of words seen n times
    self.total_words = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
        for datum in sentence.data:
            self.word_counts[datum.word] += 1
            self.total_words += 1

    for count in self.word_counts.values():
        self.inverted_count_index[count] += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    for token in sentence:
        token_count = self.word_counts[token]
        score += log(token_count + 1)
        if self.inverted_count_index[token_count + 1] == 0:
            # Good-Turing smoothing is known to have problems with high-counts words. A more
            # appropriate thing to do is to smooth the counts using a best-fit power law graph.
            score += log(self.inverted_count_index[token_count] * 1.2) - log(self.inverted_count_index[token_count])
        elif token_count > 0:
            score += log(self.inverted_count_index[token_count + 1]) - log(self.inverted_count_index[token_count])
        else:
            score += log(self.inverted_count_index[token_count + 1]) - log(self.total_words)

    return score
