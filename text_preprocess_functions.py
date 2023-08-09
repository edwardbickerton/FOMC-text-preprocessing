import re
import contractions
from unidecode import unidecode


def remove_urls(sentence):
    return re.sub(r"https?://\S+", "", sentence)


def expand_contractions(sentence):
    return contractions.fix(sentence, slang=False)


def capitalisation_normalisation(sentence):
    return sentence.casefold()


def remove_accents(sentence):
    return unidecode(sentence)


def remove_punctuation(word_list):
    return [re.sub(r"[^\w\s]", "", word) for word in word_list]


def remove_numbers(word_list):
    return [re.sub(r"\d+", "", word) for word in word_list]


def remove_empty(word_list):
    return [word for word in word_list if word != ""]
