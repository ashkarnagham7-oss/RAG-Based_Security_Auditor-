from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "vulnerable_code_combined.csv"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "securecode_unixcoder"

UNIXCODER_MODEL_NAME = "microsoft/unixcoder-base"

TOP_K = 3
OPENAI_MODEL = "gpt-5.4"