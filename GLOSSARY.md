# Glossary for NLP book
## Chapter 2

    Affixes - additional meanings of various kinds e.g. `ful` and `ly` in `carefully`
    Clitic - class of morphemes that have no standalone meaning. E.g. `'ve` in `I've`
    Code point - unique id of each character
    Derivational morphemes - Idiosyncratic and semantically harder to predict
    Glyph - visual representation of the character
    Inflectional morphemes - `s` and `es` for plural i.e. grammatically and semantically predictable. Clear syntactic role
    Morpheme - `er` is the morpheme of `longer`
    Morphology - the study of morphemes
    Utterance - linguistic term for spoken lang version of sentence
    U+ - which means: "the following is a Unicode hex representation of a code point"
## Chapter 3

    Data contamination - when some parts of the test set are present in the training set, leading to overoptimistic eval results
    Development  test set (dev set) - validation set
    Extrinsic evaluation - evaluating performance of LM by embedding in app and measuring
    Intrinsic evaluation - evaluate performance (of LM) independent of any app
    Perplexity - a common (pred accuracy) eval metric for n-gram- and Large-LM $\sqrt[n]{\frac{1}{P(w_1w_2\dots w_N}}$
    Relative frequency - frequency of a word divided by frequency of prefix (of variable  N length) that occur to that word
    Smoothing (or Discounting) - a way to deal with "zero prob n-grams", smoothing the distribution s.t. no 0 prob
## Chapter 6

    Embedding matrix - a (|V| x d) dictionary matrix of static embeddings, where d is the embedding dimension
    Pooling - combining multiple (embedding) vectors into a single vector, e.g. by averaging or maxing
## Chapter 7

    Alignment (a.k.a. preference alignment) - after pretraining and SFT alignment training ensures helpful contra harmfull behaviour
    Autoregressive generation - predicting the next token given the previous ones (repeatedly)
    Decoding - the task of choosing the next token to generate, based on the models output distribution
    Few-shot prompting - giving a few input-output examples (in the prompt) to guide the LM
    Fine tuning - training some or all parameters from pretrained model, with a domain specific dataset
    Greedy algorithm - makes the most optimal choice at each step, regardless of best choice in hindsight
    Greedy decoding - always choosing the most likely next token (simplest approach)
    Instruction tuning - training a (pretrained) LM to take instructions, a.k.a. SFT (supervised fine-tuning)
    MMLU - Massive Multitask Language Understanding benchmark, 57 subjects, multiple choice
    Teacher forcing - using true next word in sequence rather than predicted token in LLM training algo
    Zero-shot prompting - giving only the instruction (in the prompt) to guide the LM
## Chapter 17

    Closed Class (POS) - prepositions, pronouns, conjunctions etc. rarely new words added (closed membership).
    Common Nouns - Concrete terms such as cat and mango; abstractions like algorithm and beauty.
    Directional Adverbs - e.g. north, south, uphill, downstairs.
    Locative Adverbs - e.g. home, here, there, outside.
    Named Entity - roughly speaking, anything we can refer to with a proper name, e.g. person, org, location, product etc.
    NER (Named Entity Recognition) - the task of identifying and classifying named entities in text e.g. NOUN, VERB or PERSON, ORG
    Open Class (POS) - nouns, verbs, adjectives, adverbs etc. new words added frequently (open membership).
    Phrasal Verb - e.g. turn down, give up, run into. A verb combined with a preposition or adverb.
    POS (Part of Speech) - noun, verb, pronoun, preposition, adverb, conjunction, participle, articla, particle etc.
    POS Tagging - the task of assigning POS tags to each (tokenized) word in a sentence.
    Proper Nouns - Specific entities such as John, London, Linux.
