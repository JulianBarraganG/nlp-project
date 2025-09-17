import math
from typing import TypeAlias
from ngrams.utils import(
    NGramsDict,
    DataInconsistencyError
)

SeqProbDict: TypeAlias = dict[tuple[str, ...], float]

Tokens: TypeAlias = list[str]
TokenizedSentences: TypeAlias = list[Tokens]

class NGramLM:
    """Class to represent an N-gram language model.
    The model takes as input the n-grams and (n-1)-grams count dictionaries,
    and computes the conditional probabilities of the n-grams given the (n-1)-grams. 
    """
    def __init__(self, nm1grams: NGramsDict, ngrams: NGramsDict):
        self.ngrams = ngrams
        self.nm1grams = nm1grams
        self.n = len(list(ngrams.keys())[0])
        self.probabilities = {key[-1]: 0.0 for key in self.ngrams.keys()}
        self._calc_word_probabilities()

    def _calc_word_probabilities(self) -> None:
        """For word (token) in vocabulary, estimate probabilities by counts"""
        for ngram in self.ngrams.keys():
            word = ngram[-1]
            prefix = ngram[:-1]
            self.probabilities[word] = self.ngrams[ngram] / self.nm1grams[prefix]
    
    def get_sentence_probability(self, text: Tokens) -> float:
        """Get the probability of a sentence (list of tokens) under this model."""
        log_prob = 0.0
        for token in text:
            print(token)
            if token == "<s>":
                continue # skip start token
            prob = self.probabilities.get(token, 0.0)
            log_prob += math.log(prob) if prob > 0 else 0.0 # handle unknown words (bad assumption)
            # Could benefit from add-one smoothing instead of ignoring unknown words
        
        return math.exp(log_prob)