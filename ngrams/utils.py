import polars as pl
import nltk
from transformers import AutoTokenizer
from data.const import ARB_CACHE, KOR_CACHE, TELU_CACHE
from typing import TypeAlias, cast

NGram = tuple[str, ...]
Tokens: TypeAlias = list[str]
TokenizedSentences: TypeAlias = list[Tokens]
NGramsDict: TypeAlias = dict[NGram, int]
ModelReadyData: TypeAlias = tuple[tuple[NGramsDict, NGramsDict], list[list[NGram]]]

class DataInconsistencyError(Exception):
    """Custom error for data inconsistency issues."""

# Get multilingual bert tokenizer
mbert = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased")
mbert.add_tokens(["<s>", "</s>"])  # Add start and end tokens

def tokenize(text: str, tokenizer = mbert.tokenize, n=1) -> list[str]:
    """Tokenizes text ready for n-gram using the provided tokenizer function."""
    assert n >= 1, "n must be at least 1"
    pad = (n-1) if n > 1 else 1
    text = ("<s>" * pad + text + "</s>")
    return tokenizer(text)

def train_test_split_and_tokenize(corpus: pl.Series,
                                  train_split: float = 0.7,
                                  n: int = 1,
                                  verbose: bool = False,
) -> tuple[TokenizedSentences, TokenizedSentences]:
    """Split into train, val, test"""
    train_index = int(len(corpus) * train_split)
    ngram_ready_tokens = [tokenize(seq, n=n) for seq in corpus]
    context_train = ngram_ready_tokens[:train_index]
    context_test = ngram_ready_tokens[train_index:]
    if verbose:
        print(f"Train size: {len(context_train):>15,} tokens.")
        print(f"Test size: {len(context_test):>16,} tokens.")
        print(f"Total corpus size: {len(context_train) + len(context_test):>8,} tokens.")
    return context_train, context_test


def get_ngrams_dict(
    tokens: Tokens,
    n: int,
    verbose: bool = False,
) -> NGramsDict:
    """Get n-grams count dictionary from list of tokens."""
    n_grams_gen = nltk.ngrams(tokens, n)
    count_dict = {}
    num_duplicates = 0
    for gram in n_grams_gen:
        if gram in count_dict:
            count_dict[gram] += 1
            num_duplicates += 1
            if verbose and num_duplicates <= 5:
                print("Duplicate gram found: ", gram)
            if verbose and num_duplicates == 6:
                print("...")  # Indicate more duplicates exist
        else:
            count_dict[gram] = 1
    if verbose:
        print(f"Number of unique {n}-grams:  {len(count_dict):,}")
        print(f"Total number of {n}-grams:  {sum(count_dict.values()):,}")
        print(f"Number of duplicate {n}-grams encountered:  {num_duplicates:,}")
        assert num_duplicates == sum(count_dict.values()) - len(count_dict), "Duplicate count mismatch!"

    return count_dict


def get_ngrams_dict_from_sentences(
    sentences: TokenizedSentences,
    n: int,
    verbose: bool = False,
) -> NGramsDict:
    """Get n-grams count dictionary from list of sentences."""
    all_count_dict = {}
    
    for sentence in sentences:
        # Get n-grams from this padded sentence
        sentence_dict = get_ngrams_dict(sentence, n, verbose=False)
        
        # Merge into overall dictionary
        for gram, count in sentence_dict.items():
            all_count_dict[gram] = all_count_dict.get(gram, 0) + count
    if verbose:
        print(f"Number of unique {n}-grams:  {len(all_count_dict):,}")
        print(f"Total number of {n}-grams:  {sum(all_count_dict.values()):,}")
        print(f"Number of duplicate {n}-grams encountered:  {sum(all_count_dict.values()) - len(all_count_dict):,}")
    
    return all_count_dict

def get_model_ready_data(corpus: pl.Series, n: int) -> ModelReadyData:
    """One function to call on relevant series, to get NGramModel ready data"""
    x_train, x_test = train_test_split_and_tokenize(corpus, n=n)
    nm1grams_dict = get_ngrams_dict_from_sentences(x_train, n-1, verbose=False)
    ngrams_dict = get_ngrams_dict_from_sentences(x_train, n, verbose=False)
    # Make test into list of ngram sentences
    x_test = list([nltk.ngrams(sentence, n) for sentence in x_test])
    x_test = cast(list[list[NGram]], x_test)  # Type hinting for clarity
    data = cast(ModelReadyData, (nm1grams_dict, ngrams_dict, x_test))
    return data
