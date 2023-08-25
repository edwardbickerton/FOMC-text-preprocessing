import pandas as pd
from tqdm import tqdm
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

from configurations import Configuration
from preprocessing_rules import (
    remove_urls,
    capitalisation_normalisation,
    remove_accents,
    expand_contractions,
    remove_punctuation,
    remove_numbers,
    lemmatization,
)
from global_vars import ALL_DOCS_FILE_PATH, NUM_DOCS

THRESHOLDS = [0.4, 0.4, 0.4]
MIN_COUNT = 2**11 + 2**10
FREEZE = False


n_gram_config = Configuration(
    "n_gram",
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
    ],
)


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


class LoadSentence:
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for document in tqdm(
            yield_document(self.path), total=NUM_DOCS, unit="documents"
        ):
            for sentence in n_gram_config.preprocess_document_string(document):
                yield sentence


sentences = LoadSentence(ALL_DOCS_FILE_PATH)
print("TRAINING BIGRAM MODEL")
bigram_model = Phrases(
    sentences,
    min_count=MIN_COUNT,
    threshold=THRESHOLDS[0],
    delimiter="_",
    scoring="npmi",
    connector_words=ENGLISH_CONNECTOR_WORDS,
)
if FREEZE:
    bigram_model = bigram_model.freeze()
bigram_model.save("gensim_phrase_models/bigram_phrases.pkl")


class LoadBigramSentence:
    def __iter__(self):
        for sentence in LoadSentence(ALL_DOCS_FILE_PATH):
            yield bigram_model[sentence]


bigram_sentences = LoadBigramSentence()
print("TRAINING TRIGRAM MODEL")
trigram_model = Phrases(
    bigram_sentences,
    min_count=MIN_COUNT,
    threshold=THRESHOLDS[1],
    delimiter="_",
    scoring="npmi",
    connector_words=ENGLISH_CONNECTOR_WORDS,
)
if FREEZE:
    trigram_model = trigram_model.freeze()
trigram_model.save("gensim_phrase_models/trigram_phrases.pkl")


class LoadTrigramSentence:
    def __iter__(self):
        for sentence in LoadBigramSentence():
            yield trigram_model[sentence]


trigram_sentences = LoadTrigramSentence()
print("TRAINING FOURGRAM MODEL")
fourgram_model = Phrases(
    trigram_sentences,
    min_count=MIN_COUNT,
    threshold=THRESHOLDS[2],
    delimiter="_",
    scoring="npmi",
    connector_words=ENGLISH_CONNECTOR_WORDS,
)
if FREEZE:
    fourgram_model = fourgram_model.freeze()
fourgram_model.save("gensim_phrase_models/fourgram_phrases.pkl")
