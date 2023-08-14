import pandas as pd
from tqdm import tqdm

from configurations import baseline_config, lightweight_config, heavyweight_config
from global_vars import ALL_DOCS_FILE_PATH, PREPROCESSED_DATA_DIR, NUM_DOCS

if __name__ == "__main__":
    all_fomc_documents = pd.read_csv(ALL_DOCS_FILE_PATH)
    for config in [baseline_config, lightweight_config, heavyweight_config]:
        tqdm.pandas(
            total=NUM_DOCS,
            desc=f"Applying {config.name} preprocessing",
            unit="documents",
        )
        df = all_fomc_documents.copy()
        df["text"] = df["text"].progress_apply(
            lambda document: list(config.preprocess_document_string(document))
        )
        df.to_csv(f"{PREPROCESSED_DATA_DIR}{config.name}.csv", index=False)
