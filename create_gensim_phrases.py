import pandas as pd
from tqdm import tqdm
from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS

from configurations import lightweight_config
from global_vars import ALL_DOCS_FILE_PATH, NUM_DOCS

THRESHOLDS = [0.66, 0.6, 0.5]
MIN_COUNT = 3067


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
            for sentence in lightweight_config.preprocess_document_string(document):
                yield sentence


class LoadNgramSentence:
    def __init__(self, phrase_model):
        self.phrase_model = phrase_model

    def __iter__(self):
        for sentence in LoadSentence(ALL_DOCS_FILE_PATH):
            yield self.phrase_model[sentence]


print("TRAINING BIGRAM MODEL")
sentences = LoadSentence(ALL_DOCS_FILE_PATH)
bigram_model = Phrases(
    sentences,
    min_count=MIN_COUNT,
    threshold=THRESHOLDS[0],
    delimiter="_",
    scoring="npmi",
    connector_words=ENGLISH_CONNECTOR_WORDS,
)
bigram_model.save("gensim_phrase_models/bigram_phrases.pkl")


print("TRAINING TRIGRAM MODEL")
bigram_sentences = LoadNgramSentence(bigram_model)
trigram_model = Phrases(
    bigram_sentences,
    min_count=MIN_COUNT,
    threshold=THRESHOLDS[1],
    delimiter="_",
    scoring="npmi",
    connector_words=ENGLISH_CONNECTOR_WORDS,
)
trigram_model.save("gensim_phrase_models/trigram_phrases.pkl")


print("TRAINING FOURGRAM MODEL")
trigram_sentences = LoadNgramSentence(trigram_model)
fourgram_model = Phrases(
    trigram_sentences,
    min_count=MIN_COUNT,
    threshold=THRESHOLDS[2],
    delimiter="_",
    scoring="npmi",
    connector_words=ENGLISH_CONNECTOR_WORDS,
)
fourgram_model.save("gensim_phrase_models/fourgram_phrases.pkl")
