import pandas as pd
import ast

from global_vars import N_GRAM_FILE_PATH


def top_t_frequent_n_grams(t):
    try:
        n_gram_df = pd.read_csv(N_GRAM_FILE_PATH, nrows=t)

    except FileNotFoundError:
        print(
            f"Cannot find {N_GRAM_FILE_PATH}. Run create_n_grams.py to generate list of n_grams."
        )
        exit()

    return set(ast.literal_eval(string_tuple) for string_tuple in n_gram_df["n_gram"])


def min_frequency_n_grams(f):
    n_grams_set = set()

    try:
        for chunk in pd.read_csv(N_GRAM_FILE_PATH, chunksize=2**10):
            for index, row in chunk.iterrows():
                if row["frequency"] >= f:
                    string_tuple = row["n_gram"]
                    n_grams_set.add(ast.literal_eval(string_tuple))
                else:
                    return n_grams_set

    except FileNotFoundError:
        print(
            f"Cannot find {N_GRAM_FILE_PATH}. Run create_n_grams.py to generate list of n_grams."
        )
        exit()
