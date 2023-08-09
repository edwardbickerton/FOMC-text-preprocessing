import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from text_preprocess_functions import (
    remove_urls,
    capitalisation_normalisation,
    remove_accents,
    expand_contractions,
    remove_punctuation,
    remove_numbers,
    remove_empty,
)
from hidden_vars import DATA_DIR

ALL_DOCS_FILE = "fomc_documents.csv"
ALL_DOCS_FILE_PATH = DATA_DIR + ALL_DOCS_FILE


def yield_document(path):
    try:
        for chunk in pd.read_csv(path, usecols=["text"], chunksize=2**10):
            documents = chunk["text"].to_list()
            for document in documents:
                yield document

    except FileNotFoundError:
        print(f"The path provided for your dataset does not exist: {path}")
        exit()
    except ValueError:
        print("The CSV file has no 'text' column.")
        exit()


def yield_sentence(path):
    for document in yield_document(path):
        for sentence in sent_tokenize(document):
            yield sentence


def process_sentence(sentence):
    sentence = remove_urls(sentence)

    sentence = expand_contractions(sentence)

    sentence = capitalisation_normalisation(sentence)

    sentence = remove_accents(sentence)

    word_list = word_tokenize(sentence)

    word_list = remove_punctuation(word_list)

    word_list = remove_numbers(word_list)

    word_list = remove_empty(word_list)

    # MAYBE: n_grams, remove stop words, lemmatization

    return word_list


if __name__ == "__main__":
    with open("clean_sentences.txt", "a") as file:
        for sentence in yield_sentence(ALL_DOCS_FILE_PATH):
            file.write(" ".join(process_sentence(sentence)) + "\n")
