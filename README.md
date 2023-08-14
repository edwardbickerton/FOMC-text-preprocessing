# FOMC-text-preprocessing
Preprocessing the text data scraped using [Fed-Scraper](https://github.com/rw19842/Fed-Scraper).

## Usage

### 1. `hidden_vars.py`
Add a file `hidden_vars.py` containing a variable `DATA_DIR` equal to a path of the directory containing the `csv` file produced by [Fed-Scraper](https://github.com/rw19842/Fed-Scraper). This can be obtained by running the web scraper or simply downloading the dataset from [kaggle](https://www.kaggle.com/datasets/edwardbickerton/fomc-text-data).

### 2. [`configurations.py`](configurations.py)
Create configurations using the preprocessing rules found in [`preprocessing_rules.py`](preprocessing_rules.py) and add them to the `CONFIGS` list.

Note: some rules take as input a:
- `sentence` - a string containing a sentence, while others take
- `word_list` - a list of strings which are each words from a sentence.

Note: you must run [`create_n_grams.py`](create_n_grams.py) before using the `n_gram_creation` rule.

### 3. [`process_dataset.py`](process_dataset.py)
Run [`process_dataset.py`](process_dataset.py) to execute each of the configurations on the dataset, saving the results in the directory `PREPROCESSED_DATA_DIR` specified in [`global_vars.py`](global_vars.py).
