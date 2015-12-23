from collections import Counter
from math import log

class StupidBackoffLanguageModel:

  def __init__(self, corpus):
    """Initialize your data structures in the constructor."""
    self.ngrams = Counter()
    self.total_words = 0
    self.train(corpus)

  def train(self, corpus):
    """ Takes a corpus and trains your language model.
        Compute any counts or other corpus statistics in this function.
    """
    for sentence in corpus.corpus:
        for i in range(len(sentence.data)):
            cur_word = sentence.data[i].word
            self.ngrams[cur_word] += 1
            self.total_words += 1
            for j in reversed(range(i)):
                cur_word = sentence.data[j].word + ',' + cur_word
                self.ngrams[cur_word] += 1

  def score(self, sentence):
    """ Takes a list of strings as argument and returns the log-probability of the
        sentence using your language model. Use whatever data you computed in train() here.
    """
    score = 0.0
    for i in range(len(sentence)):
        cur_ngram = sentence[i]
        prev_ngram = ""
        for j in reversed(range(i)):
            if (self.ngrams[cur_ngram] == 0) or (j == 0):
                partial_score = 0.0
                if ',' in prev_ngram: # > 2-grams
                    prev_ngram_counts = self.ngrams[prev_ngram]
                    prev_minus_one_ngram_counts = self.ngrams[prev_ngram[:prev_ngram.rfind(',')]]
                    assert (prev_ngram_counts <= prev_minus_one_ngram_counts)

                    partial_score = log(0.4**j) + log(self.ngrams[prev_ngram]) - log(prev_minus_one_ngram_counts)
                elif prev_ngram != "": # Unigram
                    partial_score = log(0.4**i) + log(self.ngrams[prev_ngram]) - log(self.total_words)
                else: # Word is not found in dictionary
                    pass

                score += partial_score
                break
            prev_ngram = cur_ngram
            cur_ngram = sentence[j] + ',' + cur_ngram

    return score
