from nltk.util import everygrams
import pandas as pd
from tqdm import tqdm
import time

from configurations import Configuration
from preprocessing_rules import (
    remove_urls,
    capitalisation_normalisation,
    remove_accents,
    expand_contractions,
    remove_punctuation,
    remove_numbers,
)
from global_vars import (
    ALL_DOCS_FILE,
    ALL_DOCS_FILE_PATH,
    N_GRAM_FILE_PATH,
    NUM_DOCS,
    MAX_N_GRAM_LENGTH,
)


n_gram_config = Configuration("n_gram")
n_gram_config.rules["sentence"] = [
    remove_urls,
    expand_contractions,
    capitalisation_normalisation,
    remove_accents,
]
n_gram_config.rules["word_list"] = [
    remove_punctuation,
    remove_numbers,
]


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


if __name__ == "__main__":
    start_time = time.time()

    n_gram_frequencies = dict()
    for document in tqdm(
        yield_document(ALL_DOCS_FILE_PATH), total=NUM_DOCS, unit="documents"
    ):
        for sentence in n_gram_config.preprocess_document_string(document):
            for n_gram in everygrams(sentence, min_len=2, max_len=MAX_N_GRAM_LENGTH):
                if n_gram in n_gram_frequencies:
                    n_gram_frequencies[n_gram] += 1
                else:
                    n_gram_frequencies[n_gram] = 1

    n_gram_df = pd.DataFrame(
        list(n_gram_frequencies.items()),
        columns=["n_gram", "frequency"],
    )
    n_gram_df.sort_values(
        by="frequency", ascending=False, inplace=True, ignore_index=True
    )
    n_gram_df.to_csv(N_GRAM_FILE_PATH)

    end_time = time.time()
    print(
        f"Found {len(n_gram_df)} n_grams in {ALL_DOCS_FILE} in {end_time - start_time} seconds."
    )
