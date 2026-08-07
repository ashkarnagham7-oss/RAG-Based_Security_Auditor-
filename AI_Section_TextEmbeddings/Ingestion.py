import json
from collections import Counter
from pathlib import Path
from typing import Any

import chromadb
import torch
from FlagEmbedding import BGEM3FlagModel


# Configuration

DATASET_FILE = Path("final-knowledge-base.jsonl")

CHROMA_DIRECTORY = Path("chroma_db")
COLLECTION_NAME = "securecode_vulnerability_kb"

EMBEDDING_MODEL = "BAAI/bge-m3"

EMBEDDING_BATCH_SIZE = 8
CHROMA_BATCH_SIZE = 100


MAX_LENGTH = 512

METADATA_FIELDS = [
    "source_index",
    "language",
    "cwe",
    "category",
    "subcategory",
    "severity",
    "complexity",
    "technique",
    "cve",
    "incident_year",
    "owasp_2025",
]

# Dataset loading

def read_jsonl(file_path: Path) -> list[dict]:
    """Read and validate a JSONL dataset."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path.resolve()}"
        )

    records: list[dict] = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} does not contain a JSON object."
                )

            records.append(record)

    if not records:
        raise ValueError("The dataset contains no records.")

    return records


# Validation

def validate_records(records: list[dict]) -> None:
    """Validate IDs and required fields before creating embeddings."""

    source_indices = []

    for position, record in enumerate(records):
        source_index = record.get("source_index")
        functional_description = record.get(
            "functional_description"
        )

        if not isinstance(source_index, int):
            raise ValueError(
                f"Record at dataset position {position} has a "
                f"missing or invalid source_index."
            )

        if source_index < 0:
            raise ValueError(
                f"source_index cannot be negative: {source_index}"
            )

        if not isinstance(functional_description, str):
            raise ValueError(
                f"Record {source_index} has a missing or invalid "
                f"functional_description."
            )

        if not functional_description.strip():
            raise ValueError(
                f"Record {source_index} has an empty "
                f"functional_description."
            )

        source_indices.append(source_index)

    duplicate_indices = [
        value
        for value, count in Counter(source_indices).items()
        if count > 1
    ]

    if duplicate_indices:
        raise ValueError(
            "Duplicate source_index values found: "
            + ", ".join(
                str(value)
                for value in sorted(duplicate_indices)
            )
        )

    expected_indices = list(range(len(records)))
    actual_indices = sorted(source_indices)

    if actual_indices != expected_indices:
        missing_indices = sorted(
            set(expected_indices) - set(actual_indices)
        )

        unexpected_indices = sorted(
            set(actual_indices) - set(expected_indices)
        )

        raise ValueError(
            "source_index values are unique but not sequential.\n"
            f"Expected range: 0-{len(records) - 1}\n"
            f"Missing indices: {missing_indices[:20]}\n"
            f"Unexpected indices: {unexpected_indices[:20]}"
        )


# Metadata preparation
def is_valid_metadata_value(value: Any) -> bool:
    """
    Chroma metadata supports scalar strings, integers,
    floats, booleans, and arrays of those types.
    """

    if isinstance(value, (str, int, float, bool)):
        return True

    if isinstance(value, list):
        return all(
            isinstance(item, (str, int, float, bool))
            for item in value
        )

    return False


def build_metadata(record: dict) -> dict:
    """
    Copy selected metadata fields.

    Null values are omitted because they are not useful for filtering.
    """

    metadata = {}

    for field in METADATA_FIELDS:
        value = record.get(field)

        if value is None:
            continue

        if not is_valid_metadata_value(value):
            raise ValueError(
                f"Unsupported metadata value for field '{field}' "
                f"in record {record.get('source_index')}: "
                f"{type(value).__name__}"
            )

        metadata[field] = value

    return metadata


# Model loading

def load_embedding_model() -> BGEM3FlagModel:
    """Load BGE-M3 using GPU when CUDA is available."""

    cuda_available = torch.cuda.is_available()

    print("\nLoading embedding model")
    print("-" * 60)
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("Using FP16: yes")
    else:
        print("Device: CPU")
        print("Using FP16: no")

    model = BGEM3FlagModel(
        EMBEDDING_MODEL,
        use_fp16=cuda_available,
    )

    return model


# Chroma collection

def create_fresh_collection(
    client: chromadb.PersistentClient,
):
    """
    Delete an old collection with the same name and create
    a new cosine-distance collection.
    """

    existing_names = {
        collection.name
        for collection in client.list_collections()
    }

    if COLLECTION_NAME in existing_names:
        print(
            f"Deleting existing collection: {COLLECTION_NAME}"
        )
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
        metadata={
            "embedding_model": EMBEDDING_MODEL,
            "embedded_field": "functional_description",
            "id_field": "source_index",
        },
    )

    return collection


# Ingestion

def ingest_records(
    records: list[dict],
    model: BGEM3FlagModel,
    collection,
) -> None:
    """Generate dense embeddings and store them in Chroma."""

    total_records = len(records)

    for start in range(0, total_records, CHROMA_BATCH_SIZE):
        end = min(
            start + CHROMA_BATCH_SIZE,
            total_records,
        )

        batch_records = records[start:end]

        ids = [
            str(record["source_index"])
            for record in batch_records
        ]

        documents = [
            record["functional_description"].strip()
            for record in batch_records
        ]

        metadatas = [
            build_metadata(record)
            for record in batch_records
        ]

        print(
            f"Embedding and storing records "
            f"{start + 1}-{end} of {total_records}..."
        )

        model_output = model.encode(
            documents,
            batch_size=EMBEDDING_BATCH_SIZE,
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )

        dense_vectors = model_output["dense_vecs"]

        if len(dense_vectors) != len(batch_records):
            raise RuntimeError(
                "The number of embeddings does not match "
                "the number of records in the batch."
            )

        embeddings = dense_vectors.tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )


# Verification

def verify_collection(
    records: list[dict],
    collection,
) -> None:
    """Verify that every dataset record was stored."""

    expected_count = len(records)
    stored_count = collection.count()

    print("\nIngestion verification")
    print("-" * 60)
    print(f"Dataset records: {expected_count:,}")
    print(f"Chroma records:  {stored_count:,}")

    if stored_count != expected_count:
        raise RuntimeError(
            "Chroma record count does not match the dataset count."
        )

    first_record = records[0]
    first_id = str(first_record["source_index"])

    stored_record = collection.get(
        ids=[first_id],
        include=[
            "documents",
            "metadatas",
        ],
    )

    if not stored_record["ids"]:
        raise RuntimeError(
            f"Verification record {first_id} was not found."
        )

    print("\nExample stored record")
    print("-" * 60)
    print(f"Chroma ID: {stored_record['ids'][0]}")
    print(
        "Document:",
        stored_record["documents"][0],
    )
    print(
        "Metadata:",
        json.dumps(
            stored_record["metadatas"][0],
            ensure_ascii=False,
            indent=2,
        ),
    )


# Test semantic query

def run_test_query(
    model: BGEM3FlagModel,
    collection,
) -> None:
    """Run one test query to confirm semantic retrieval works."""

    query_text = (
        "Handle an HTTP GET request that reads a theme-specific preview HTML file from"
        "storage based on a request parameter and returns the file contents in the response."
    )

    query_output = model.encode(
        [query_text],
        batch_size=1,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    query_embedding = (
        query_output["dense_vecs"][0]
        .tolist()
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    print("\nTest query")
    print("-" * 60)
    print(query_text)

    print("\nTop retrieved records")
    print("-" * 60)

    for rank, (
        chroma_id,
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        similarity = 1.0 - distance

        print(f"\nRank {rank}")
        print(f"Chroma ID: {chroma_id}")
        print(f"Similarity: {similarity:.4f}")
        print(f"CWE: {metadata.get('cwe')}")
        print(
            f"Subcategory: "
            f"{metadata.get('subcategory')}"
        )
        print(f"Description: {document}")


# Main

def main() -> None:
    print("=" * 60)
    print("SECURECODE KNOWLEDGE-BASE INGESTION")
    print("=" * 60)

    records = read_jsonl(DATASET_FILE)

    print(f"Dataset: {DATASET_FILE.resolve()}")
    print(f"Records loaded: {len(records):,}")

    validate_records(records)

    print("Dataset validation: passed")
    print(
        f"source_index range: "
        f"0-{len(records) - 1}"
    )

    model = load_embedding_model()

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIRECTORY)
    )

    collection = create_fresh_collection(client)

    ingest_records(
        records=records,
        model=model,
        collection=collection,
    )

    verify_collection(
        records=records,
        collection=collection,
    )

    run_test_query(
        model=model,
        collection=collection,
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(
        f"Chroma directory: "
        f"{CHROMA_DIRECTORY.resolve()}"
    )
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("Embedded field: functional_description")


if __name__ == "__main__":
    main()
