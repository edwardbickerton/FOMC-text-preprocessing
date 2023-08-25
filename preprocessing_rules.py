import re
import contractions
from unidecode import unidecode
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.util import everygrams
from gensim.models.phrases import Phrases

from n_grams import min_frequency_n_grams
from global_vars import MAX_N_GRAM_LENGTH

STOPWORDS = set(stopwords.words("english"))
N_GRAMS = None
GENSIM_PHRASES = None


def remove_urls(sentence):
    return re.sub(r"https?://\S+", "", sentence)


def expand_contractions(sentence):
    return contractions.fix(sentence, slang=False)


def capitalisation_normalisation(sentence):
    return sentence.casefold()


def remove_accents(sentence):
    return unidecode(sentence)


def remove_punctuation(word_list):
    new_word_list = []
    for word in word_list:
        new_word = re.sub(r"[^\w\s]", "", word)
        if new_word != "":
            new_word_list.append(new_word)
    return new_word_list


def remove_numbers(word_list):
    new_word_list = []
    for word in word_list:
        new_word = re.sub(r"\d+", "", word)
        if new_word != "":
            new_word_list.append(new_word)
    return new_word_list


def remove_stopwords(word_list):
    return [word for word in word_list if word not in STOPWORDS]


def remove_short_words(word_list, threshold=2):
    return [word for word in word_list if len(word) > threshold]


def lemmatization(word_list):
    lemma_list = []
    pos_list = pos_tag(word_list, tagset="universal", lang="eng")

    for pair in pos_list:
        word = pair[0]
        tag = pair[1]

        if tag == "NOUN":
            lemma = WordNetLemmatizer().lemmatize(word, "n")
        elif tag == "VERB":
            lemma = WordNetLemmatizer().lemmatize(word, "v")
        elif tag == "ADJ":
            lemma = WordNetLemmatizer().lemmatize(word, "a")
        elif tag == "ADV":
            lemma = WordNetLemmatizer().lemmatize(word, "r")
        else:
            lemma = word

        if lemma != "":
            lemma_list.append(lemma)

    return lemma_list


def n_gram_creation(word_list):
    global N_GRAMS
    if N_GRAMS is None:
        N_GRAMS = min_frequency_n_grams(2**10)

    sentence_n_grams = set(
        n_gram
        for n_gram in everygrams(
            word_list,
            min_len=2,
            max_len=MAX_N_GRAM_LENGTH,
        )
        if n_gram in N_GRAMS
    )

    if len(sentence_n_grams) == 0:
        return word_list

    match_n_gram = sorted(list(sentence_n_grams), key=len).pop()

    result = []
    i = 0
    while i < len(word_list):
        merged_word = word_list[i]
        for word in match_n_gram:
            if merged_word == word:
                j = i + 1
                while j < len(word_list) and word_list[j] in match_n_gram:
                    merged_word += "_" + word_list[j]
                    j += 1
                i = j - 1
                break

        result.append(merged_word)
        i += 1

    return n_gram_creation(result)


def gensim_phrase(word_list):
    global GENSIM_PHRASES

    if GENSIM_PHRASES is None:
        GENSIM_PHRASES = [
            Phrases.load(f"gensim_phrase_models/{n}gram_phrases.pkl")
            for n in ["bi", "tri", "four"]
        ]

    n_gram_word_list = word_list
    for phrase_model in GENSIM_PHRASES:
        n_gram_word_list = phrase_model[n_gram_word_list]

    return n_gram_word_list
