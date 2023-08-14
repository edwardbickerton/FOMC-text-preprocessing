import pandas as pd
from tqdm import tqdm
import os.path
from os import mkdir

from configurations import CONFIGS
from global_vars import (
    ALL_DOCS_FILE,
    ALL_DOCS_FILE_PATH,
    PREPROCESSED_DATA_DIR,
    NUM_DOCS,
    SUB_FILES,
)

if __name__ == "__main__":
    if not os.path.isdir(PREPROCESSED_DATA_DIR):
        mkdir(PREPROCESSED_DATA_DIR)

    all_fomc_documents = pd.read_csv(ALL_DOCS_FILE_PATH)
    for config in CONFIGS:
        config_dir = os.path.join(PREPROCESSED_DATA_DIR, config.name)
        if not os.path.isdir(config_dir):
            mkdir(config_dir)

        tqdm.pandas(
            total=NUM_DOCS,
            desc=f"Applying {config.name} preprocessing",
            unit="documents",
        )
        processed_fomc_documents = all_fomc_documents.copy()
        processed_fomc_documents["text"] = processed_fomc_documents[
            "text"
        ].progress_apply(
            lambda document: list(config.preprocess_document_string(document))
        )
        processed_fomc_documents.to_csv(
            os.path.join(config_dir, ALL_DOCS_FILE),
            index=False,
        )

        sub_data_dir = os.path.join(config_dir, "documents_by_type/")
        if not os.path.isdir(sub_data_dir):
            mkdir(sub_data_dir)

        for file in SUB_FILES:
            df = processed_fomc_documents[
                processed_fomc_documents["document_kind"].isin(file["document_kinds"])
            ].copy()

            df.sort_values(by="meeting_date", inplace=True, na_position="first")
            df.drop_duplicates(subset="url", keep="last", inplace=True)
            df.to_csv(
                os.path.join(sub_data_dir, file["name"]),
                index=False,
            )
