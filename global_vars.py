import pandas as pd
from hidden_vars import DATA_DIR

ALL_DOCS_FILE = "fomc_documents.csv"
ALL_DOCS_FILE_PATH = DATA_DIR + ALL_DOCS_FILE
PREPROCESSED_DATA_DIR = "preprocessed_data/"
MAX_N_GRAM_LENGTH = 4
N_GRAM_FILE_PATH = "n_grams.csv"
NUM_DOCS = len(pd.read_csv(ALL_DOCS_FILE_PATH))
