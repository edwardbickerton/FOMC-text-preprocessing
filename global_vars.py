import pandas as pd
from hidden_vars import DATA_DIR

ALL_DOCS_FILE = "fomc_documents.csv"
ALL_DOCS_FILE_PATH = DATA_DIR + ALL_DOCS_FILE
PREPROCESSED_DATA_DIR = "preprocessed_data/"
MAX_N_GRAM_LENGTH = 4
N_GRAM_FILE_PATH = "n_grams.csv"
NUM_DOCS = len(pd.read_csv(ALL_DOCS_FILE_PATH))
SUB_FILES = [
    {
        "name": "meeting_transcripts.csv",
        "document_kinds": ["transcript"],
    },
    {
        "name": "meeting_minutes.csv",
        "document_kinds": [
            "minutes",
            "minutes_of_actions",
            "record_of_policy_actions",
            "memoranda_of_discussion",
            "historical_minutes",
            "intermeeting_executive_committee_minutes",
        ],
    },
    {
        "name": "press_conference_transcript.csv",
        "document_kinds": ["press_conference"],
    },
    {
        "name": "policy_statements.csv",
        "document_kinds": ["statement", "implementation_note"],
    },
    {
        "name": "agendas.csv",
        "document_kinds": ["agenda"],
    },
    {
        "name": "greenbooks.csv",
        "document_kinds": [
            "greenbook",
            "greenbook_part_one",
            "greenbook_part_two",
            "greenbook_supplement",
            "tealbook_a",
        ],
    },
    {
        "name": "bluebooks.csv",
        "document_kinds": ["bluebook", "tealbook_b"],
    },
    {
        "name": "redbooks.csv",
        "document_kinds": ["redbook", "beige_book"],
    },
]
all_fomc_documents = pd.read_csv(ALL_DOCS_FILE_PATH)
non_misc_document_kinds = []
for file in SUB_FILES:
    non_misc_document_kinds += file["document_kinds"]
misc_document_kinds = [
    document_kind
    for document_kind in set(all_fomc_documents["document_kind"])
    if document_kind not in non_misc_document_kinds
]
SUB_FILES.append({"name": "miscellaneous.csv", "document_kinds": misc_document_kinds})
