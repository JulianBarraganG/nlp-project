import math
import numpy as np
from typing import TypeAlias, overload, Union
from ngrams.utils import NGramsDict

NGram = tuple[str, ...]
Tokens: TypeAlias = list[str]
TokenizedSentences: TypeAlias = list[Tokens]

class NGramLM:
    """Class to represent an N-gram language model.
    The model takes as input the n-grams and (n-1)-grams count dictionaries,
    and computes the conditional probabilities of the n-grams given the (n-1)-grams. 
    """
    def __init__(self, nm1grams: NGramsDict, ngrams: NGramsDict,
                 vocabulary: set[str] | None = None, smoothing: str | None ="laplace"):
        self.ngrams = ngrams
        self.nm1grams = nm1grams
        self.n = len(list(ngrams.keys())[0])
        if self.n <= 1:
            raise ValueError("'n' for n-grams must be greater than 1.")
        if not vocabulary:
            self.vocabulary = {token for nm1gram in self.nm1grams.keys() for token in nm1gram}
        else:
            self.vocabulary = vocabulary
        self.vocab_size = len(self.vocabulary)
        # Probabilities of words given their (n-1) prefix
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        # P(w_n | w_1 ... w_(n-1)) for each n-gram
        self.probabilities = {key: {} for key in self.nm1grams.keys()}
        self.smoothing = smoothing
        self.alpha = 1 if smoothing else 0.0
        self._calc_word_probabilities()

    def _calc_word_probabilities(self) -> None:
        """Given an (n-1)gram we want the probability distribution over every possible next word""" 
        for ngram in self.ngrams.keys():
            word = ngram[-1]
            prefix = ngram[:-1]
            invalid_nm1gram = ("<s>",) * (self.n - 1) if self.n > 2 else None
            if prefix == invalid_nm1gram:
                continue # n*start token only exists for ngrams, not (n-1)grams
            word_idx = self.word_to_idx[word]
            ngram_count = self.ngrams.get(ngram, 0) 
            nm1gram_count = self.nm1grams.get(prefix, 0)
            if self.smoothing == "laplace":
                ngram_count += self.alpha
                nm1gram_count += self.alpha * self.vocab_size
            self.probabilities[prefix][word_idx] = ngram_count / nm1gram_count
    
    def get_word_probability(self, nm1gram: NGram, word: str) -> float:
        """Get the probability of a word given its (n-1)-gram prefix."""
        if len(nm1gram) != self.n - 1:
            raise ValueError(f"Key must be of length {self.n - 1} i.e. an (n-1)-gram.")
        if nm1gram not in self.probabilities:
            raise KeyError(f"(n-1)-gram {nm1gram} not found in model.")
        if word not in self.word_to_idx:
            raise KeyError(f"Word {word} not found in vocabulary.")
        word_idx = self.word_to_idx[word]
        return self.probabilities[nm1gram][word_idx]

    def get_word_distribution(self, nm1gram: NGram) -> dict[int, float]:
        """Get the probability distribution for a given (n-1)-gram."""
        if len(nm1gram) != self.n - 1:
            raise ValueError(f"Key must be of length {self.n - 1} i.e. an (n-1)-gram.")
        if nm1gram not in self.probabilities:
            raise KeyError(f"(n-1)-gram {nm1gram} not found in model.")
        return self.probabilities[nm1gram]

    def get_sentence_probability(self, sentence: list[NGram], verbose=False) -> float:
        """Get the probability of a sentence (list of tokens) under this model."""
        log_prob = 0.0
        assert len(sentence[0]) == self.n, "Each ngram in sentence must be of length n"
        for ngram in sentence:
            word = ngram[-1]
            nm1gram = ngram[:-1]
            word_idx = self.word_to_idx.get(word, self.vocab_size + 1)
            prob_dict = self.probabilities.get(nm1gram, {})
            default_prob = self.alpha / (self.alpha * self.vocab_size) if self.smoothing else 0.0
            prob = prob_dict.get(word_idx, default_prob)
            if verbose:
                print(f"P({ngram[-1]}|{ngram[:-1]}) = {prob:.2f}")
            if self.smoothing:
                assert prob > 0.0, "Smoothing method should eliminate zero probabilities"
            if prob == 0.0 and verbose:
                print(f"WARNING: Zero probability for ngram {ngram}.")
            log_prob += math.log(prob) if prob > 0.0 else 0.0
    
        return math.exp(log_prob)

    def get_perplexity(self, sentences: list[list[NGram]]) -> float:
        """Get the perplexity of the model, on an unseen test set"""
        # Get the total number of words. Counting </s> but not <s>:
        N = sum([len(sentence) for sentence in sentences])
        full_document_prob = sum([self.get_sentence_probability(sentence) for sentence in sentences])
        if full_document_prob == 0.0:
            print("WARNING: Zero document probability.")
            return 0.0
        return math.pow(full_document_prob, -(1/N))

    @overload
    def __getitem__(self, key: NGram) -> np.ndarray:
        """Get the probability distribution for a given (n-1)-gram."""
        ...
    
    @overload
    def __getitem__(self, key: tuple[NGram, str]) -> float:
        """Get the probability of a word given its (n-1)-gram prefix."""
        ...
    
    def __getitem__(self, key: Union[NGram, tuple[NGram, str]]) -> Union[np.ndarray, float]:
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[1], str):
            nm1gram, word = key
            return self.get_word_probability(nm1gram, word) # type: ignore
        else:
            return self.get_word_distribution(key) # type: ignore