from nltk.tokenize import sent_tokenize, word_tokenize
from preprocessing_rules import (
    remove_urls,
    capitalisation_normalisation,
    remove_accents,
    expand_contractions,
    remove_punctuation,
    remove_numbers,
    remove_stopwords,
    remove_short_words,
    lemmatization,
    n_gram_creation,
)


class Configuration:
    def __init__(self, name, sentence_rules=[], word_list_rules=[]):
        self.name = name
        self.sentence_rules = sentence_rules
        self.word_list_rules = word_list_rules

    def sentences(self, string):
        return sent_tokenize(string)

    def preprocess_sentence(self, sentence):
        for sentence_rule in self.sentence_rules:
            sentence = sentence_rule(sentence)
        word_list = word_tokenize(sentence)
        word_list = [word for word in word_list if word != ""]
        return word_list

    def preprocess_word_list(self, word_list):
        for word_list_rule in self.word_list_rules:
            word_list = word_list_rule(word_list)
        return word_list

    def preprocess_document_string(self, string):
        for sentence in self.sentences(string):
            word_list = self.preprocess_sentence(sentence)
            word_list = self.preprocess_word_list(word_list)
            if word_list != []:
                yield word_list


basic_tokenizer_config = Configuration("basic_tokenizer")

baseline_config = Configuration(
    "baseline",
    word_list_rules=[
        remove_punctuation,
    ],
)

lightweight_config = Configuration(
    "lightweight",
    sentence_rules=[
        remove_urls,
        expand_contractions,
        capitalisation_normalisation,
        remove_accents,
    ],
    word_list_rules=[
        remove_punctuation,
        remove_numbers,
        remove_stopwords,
    ],
)

heavyweight_config = Configuration(
    "heavyweight",
    sentence_rules=[
        remove_urls,
        expand_contractions,
        capitalisation_normalisation,
        remove_accents,
    ],
    word_list_rules=[
        remove_punctuation,
        remove_numbers,
        lemmatization,
        remove_stopwords,
        remove_short_words,
        n_gram_creation,
    ],
)


CONFIGS = [
    basic_tokenizer_config,
    baseline_config,
    lightweight_config,
    heavyweight_config,
]
