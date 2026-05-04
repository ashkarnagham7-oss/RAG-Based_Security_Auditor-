import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma

from config import DATA_PATH, CHROMA_DIR, COLLECTION_NAME, UNIXCODER_MODEL_NAME
from embeddings_unixcoder import UniXcoderEmbeddings


def _safe_value(value):
    return None if pd.isna(value) else value


def load_documents_from_csv(csv_path=DATA_PATH):
    df = pd.read_csv(csv_path)

    required_columns = [
        "example_id",
        "cwe_id",
        "owasp_category",
        "language",
        "cve_id",
        "cvss",
        "incident_description",
        "vulnerable_code",
    ]

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["vulnerable_code"]).copy()
    df["vulnerable_code"] = df["vulnerable_code"].astype(str)

    documents = []
    ids = []

    for _, row in df.iterrows():
        example_id = str(row["example_id"])

        doc = Document(
            page_content=row["vulnerable_code"].strip(),
            metadata={
                "example_id": example_id,
                "cwe_id": _safe_value(row["cwe_id"]),
                "owasp_category": _safe_value(row["owasp_category"]),
                "language": _safe_value(row["language"]),
                "cve_id": _safe_value(row["cve_id"]),
                "cvss": _safe_value(row["cvss"]),
                "incident_description": _safe_value(row["incident_description"]),
            },
        )
        documents.append(doc)
        import uuid
        ids.append(str(uuid.uuid4()))

    return documents, ids


def build_or_replace_chroma():
    embedding_fn = UniXcoderEmbeddings(model_name=UNIXCODER_MODEL_NAME)
    documents, ids = load_documents_from_csv()

    print(f"[INFO] Loaded {len(documents)} documents from CSV")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"}
    )

    # optional cleanup if you want a fresh rebuild each time
    existing = vectorstore.get()
    existing_ids = existing.get("ids", [])
    if existing_ids:
        print(f"[INFO] Deleting existing {len(existing_ids)} records from collection")
        vectorstore.delete(ids=existing_ids)

    print("[INFO] Adding documents to Chroma...")
    vectorstore.add_documents(documents=documents, ids=ids)

    print("[INFO] Ingestion complete.")
    print(f"[INFO] Chroma DB persisted at: {CHROMA_DIR}")

    return vectorstore


if __name__ == "__main__":
    build_or_replace_chroma()