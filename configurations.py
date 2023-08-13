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
    def __init__(self, name, rules={"sentence": [], "word_list": []}):
        self.name = name
        self.rules = rules

    def sentences(self, string):
        return sent_tokenize(string)

    def preprocess_sentence(self, sentence):
        for sentence_rule in self.rules["sentence"]:
            sentence = sentence_rule(sentence)
        word_list = word_tokenize(sentence)
        word_list = [word for word in word_list if word != ""]
        return word_list

    def preprocess_word_list(self, word_list):
        for word_list_rule in self.rules["word_list"]:
            word_list = word_list_rule(word_list)
        return word_list

    def preprocess_document_string(self, string):
        for sentence in self.sentences(string):
            word_list = self.preprocess_sentence(sentence)
            word_list = self.preprocess_word_list(word_list)
            if word_list != []:
                yield word_list


baseline_config = Configuration("baseline")
baseline_config.rules["word_list"] = [
    remove_punctuation,
]

lightweight_config = Configuration("lightweight")
lightweight_config.rules["sentence"] = [
    remove_urls,
    expand_contractions,
    capitalisation_normalisation,
    remove_accents,
]
lightweight_config.rules["word_list"] = [
    remove_punctuation,
    remove_numbers,
    remove_stopwords,
]

heavyweight_config = Configuration("heavyweight")
heavyweight_config.rules["sentence"] = [
    remove_urls,
    expand_contractions,
    capitalisation_normalisation,
    remove_accents,
]
heavyweight_config.rules["word_list"] = [
    remove_punctuation,
    remove_numbers,
    remove_stopwords,
    remove_short_words,
    n_gram_creation,
    lemmatization,
]
