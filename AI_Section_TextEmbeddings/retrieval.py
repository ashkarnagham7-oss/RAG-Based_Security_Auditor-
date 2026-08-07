from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import chromadb
import torch
from FlagEmbedding import BGEM3FlagModel


# Configuration
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent


PROCESSED_PROJECT_NAME = "testcode"

METHOD_DATA_FILE = (
    PROJECT_ROOT
    / "processed"
    / PROCESSED_PROJECT_NAME
    / "method_data_owasp_post_only.csv"
)

KNOWLEDGE_BASE_FILE = (
    PROJECT_ROOT
    / "final-knowledge-base.jsonl"
)

CHROMA_DIRECTORY = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "securecode_vulnerability_kb"
EMBEDDING_MODEL = "BAAI/bge-m3"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "processed"
    / PROCESSED_PROJECT_NAME
    / "retrieval_results.jsonl"
)

SKIPPED_FUNCTIONS_FILE = (
    PROJECT_ROOT
    / "processed"
    / PROCESSED_PROJECT_NAME
    / "retrieval_skipped.csv"
)

# Retrieve three candidates for every function.
TOP_K = 3

# Number of user descriptions embedded together.
EMBEDDING_BATCH_SIZE = 8

MAX_LENGTH = 512


# File validation


def require_file(file_path: Path, description: str) -> None:
    """Raise a clear error when an input file is missing."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n"
            f"{file_path.resolve()}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"{description} is not a file:\n"
            f"{file_path.resolve()}"
        )


def require_directory(
    directory_path: Path,
    description: str,
) -> None:
    """Raise a clear error when an input directory is missing."""

    if not directory_path.exists():
        raise FileNotFoundError(
            f"{description} was not found:\n"
            f"{directory_path.resolve()}"
        )

    if not directory_path.is_dir():
        raise ValueError(
            f"{description} is not a directory:\n"
            f"{directory_path.resolve()}"
        )


# Read user functions
def read_method_data(file_path: Path) -> list[dict[str, str]]:
    """Read functions produced by users_code_preprocessing.py."""

    required_fields = {
        "method_id",
        "unit_type",
        "file_path",
        "language",
        "class_name",
        "name",
        "start_line",
        "end_line",
        "source_code",
        "functional_description",
    }

    methods: list[dict[str, str]] = []

    with file_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(
                f"The CSV has no header: {file_path.resolve()}"
            )

        missing_fields = required_fields - set(reader.fieldnames)

        if missing_fields:
            raise ValueError(
                "method_data.csv is missing required columns: "
                + ", ".join(sorted(missing_fields))
            )

        for row_number, row in enumerate(reader, start=2):
            method_id = (row.get("method_id") or "").strip()
            function_name = (row.get("name") or "").strip()
            source_code = row.get("source_code") or ""

            if not method_id:
                raise ValueError(
                    f"Missing method_id on CSV row {row_number}."
                )

            if not function_name:
                raise ValueError(
                    f"Missing function name on CSV row {row_number}."
                )

            if not source_code.strip():
                raise ValueError(
                    f"Missing source code for '{function_name}' "
                    f"on CSV row {row_number}."
                )

            methods.append(row)

    if not methods:
        raise ValueError(
            f"No functions were found in {file_path.resolve()}."
        )

    return methods


# Read full knowledge-base records


def read_knowledge_base(
    file_path: Path,
) -> dict[int, dict[str, Any]]:
    """
    Load full JSONL records and index them by source_index.

    Chroma only contains descriptions and selected metadata.
    This dictionary is used to recover vulnerable_code,
    secure_code, attack information, and all other fields.
    """

    records_by_index: dict[int, dict[str, Any]] = {}

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on knowledge-base line "
                    f"{line_number}: {error}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Knowledge-base line {line_number} "
                    f"is not a JSON object."
                )

            source_index = record.get("source_index")

            if not isinstance(source_index, int):
                raise ValueError(
                    f"Missing or invalid source_index on "
                    f"knowledge-base line {line_number}."
                )

            if source_index in records_by_index:
                raise ValueError(
                    f"Duplicate source_index found in knowledge base: "
                    f"{source_index}"
                )

            records_by_index[source_index] = record

    if not records_by_index:
        raise ValueError("The knowledge-base JSONL file is empty.")

    return records_by_index


# BGE-M3

def load_embedding_model() -> BGEM3FlagModel:
    

    cuda_available = torch.cuda.is_available()

    print("\nLoading embedding model")
    print("-" * 70)
    print(f"Model: {EMBEDDING_MODEL}")
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("Using FP16: yes")
    else:
        print("Device: CPU")
        print("Using FP16: no")

    return BGEM3FlagModel(
        EMBEDDING_MODEL,
        use_fp16=cuda_available,
    )


def embed_descriptions(
    model: BGEM3FlagModel,
    descriptions: list[str],
) -> list[list[float]]:
    """Generate dense BGE-M3 embeddings."""

    if not descriptions:
        return []

    output = model.encode(
        descriptions,
        batch_size=EMBEDDING_BATCH_SIZE,
        max_length=MAX_LENGTH,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    dense_vectors = output.get("dense_vecs")

    if dense_vectors is None:
        raise RuntimeError(
            "BGE-M3 did not return dense_vecs."
        )

    if len(dense_vectors) != len(descriptions):
        raise RuntimeError(
            "The number of embeddings does not match "
            "the number of descriptions."
        )

    return dense_vectors.tolist()


# Chroma

def open_chroma_collection():
    """
    Open the existing collection.

    This function does not delete, create, or rebuild the
    knowledge base.
    """

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIRECTORY)
    )

    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )
    except Exception as error:
        raise RuntimeError(
            f"Could not open Chroma collection "
            f"'{COLLECTION_NAME}'.\n"
            f"Run the ingestion script successfully first.\n"
            f"Original error: {error}"
        ) from error

    stored_count = collection.count()

    if stored_count == 0:
        raise RuntimeError(
            f"Chroma collection '{COLLECTION_NAME}' is empty."
        )

    print("\nChroma collection")
    print("-" * 70)
    print(f"Directory: {CHROMA_DIRECTORY.resolve()}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored knowledge records: {stored_count}")

    return collection



# Retrieval helpers


def parse_integer(
    value: str,
    field_name: str,
    method_id: str,
) -> int:
    """Convert a CSV number to an integer safely."""

    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid {field_name} for method '{method_id}': "
            f"{value!r}"
        ) from error


def cosine_similarity_from_distance(distance: float) -> float:
    """
    Convert the Chroma cosine distance used by this collection
    to an easier-to-read similarity value.
    """

    similarity = 1.0 - float(distance)

    # Protect against insignificant floating-point overflow.
    return max(-1.0, min(1.0, similarity))


def build_user_function_record(
    method: dict[str, str],
) -> dict[str, Any]:
    """Convert a CSV method row into a structured JSON record."""

    method_id = method["method_id"].strip()

    return {
        "method_id": method_id,
        "unit_type": method["unit_type"].strip(),
        "file_path": method["file_path"].strip(),
        "language": method["language"].strip(),
        "class_name": (
            method["class_name"].strip() or None
        ),
        "function_name": method["name"].strip(),
        "start_line": parse_integer(
            method["start_line"],
            "start_line",
            method_id,
        ),
        "end_line": parse_integer(
            method["end_line"],
            "end_line",
            method_id,
        ),
        "source_code": method["source_code"],
        "functional_description": (
            method["functional_description"].strip()
        ),
        "references": (
            method.get("references", "").strip()
        ),
    }


def build_candidate(
    rank: int,
    chroma_id: str,
    distance: float,
    chroma_document: str,
    chroma_metadata: dict[str, Any],
    records_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Combine the Chroma match with its full JSONL record."""

    try:
        source_index = int(chroma_id)
    except ValueError as error:
        raise ValueError(
            f"Chroma returned a non-integer ID: {chroma_id!r}"
        ) from error

    full_record = records_by_index.get(source_index)

    if full_record is None:
        raise KeyError(
            f"Chroma returned source_index {source_index}, "
            f"but it does not exist in "
            f"{KNOWLEDGE_BASE_FILE.name}."
        )

    similarity = cosine_similarity_from_distance(distance)

    return {
        "rank": rank,
        "similarity": round(similarity, 6),
        "distance": round(float(distance), 6),
        "source_index": source_index,

        # The complete record is included so that the next
        # reasoning stage has vulnerable code, secure code,
        # CWE information, causes, impacts, and other fields.
        "knowledge_record": full_record,

        # These two fields make it easier to verify that Chroma
        # and the JSONL file are synchronized.
        "retrieved_document": chroma_document,
        "retrieved_metadata": chroma_metadata,
    }



# Retrieval

def retrieve_for_methods(
    methods: list[dict[str, str]],
    model: BGEM3FlagModel,
    collection,
    records_by_index: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """
    Retrieve top-k vulnerability records for every function
    with a non-empty functional description.
    """

    valid_methods: list[dict[str, str]] = []
    skipped_methods: list[dict[str, str]] = []

    for method in methods:
        description = (
            method.get("functional_description") or ""
        ).strip()

        if not description:
            skipped_methods.append({
                "method_id": method.get("method_id", ""),
                "file_path": method.get("file_path", ""),
                "function_name": method.get("name", ""),
                "start_line": method.get("start_line", ""),
                "end_line": method.get("end_line", ""),
                "reason": "Empty functional_description",
            })
            continue

        valid_methods.append(method)

    if not valid_methods:
        raise ValueError(
            "None of the functions has a functional description."
        )

    retrieval_results: list[dict[str, Any]] = []

    total_methods = len(valid_methods)

    for batch_start in range(
        0,
        total_methods,
        EMBEDDING_BATCH_SIZE,
    ):
        batch_end = min(
            batch_start + EMBEDDING_BATCH_SIZE,
            total_methods,
        )

        batch_methods = valid_methods[batch_start:batch_end]

        batch_descriptions = [
            method["functional_description"].strip()
            for method in batch_methods
        ]

        print(
            f"\nEmbedding user functions "
            f"{batch_start + 1}-{batch_end} "
            f"of {total_methods}..."
        )

        query_embeddings = embed_descriptions(
            model,
            batch_descriptions,
        )

        query_results = collection.query(
            query_embeddings=query_embeddings,
            n_results=TOP_K,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        result_ids = query_results.get("ids")
        result_documents = query_results.get("documents")
        result_metadatas = query_results.get("metadatas")
        result_distances = query_results.get("distances")

        if not all([
            result_ids is not None,
            result_documents is not None,
            result_metadatas is not None,
            result_distances is not None,
        ]):
            raise RuntimeError(
                "Chroma returned an incomplete query response."
            )

        for batch_position, method in enumerate(batch_methods):
            user_function = build_user_function_record(method)

            ids = result_ids[batch_position]
            documents = result_documents[batch_position]
            metadatas = result_metadatas[batch_position]
            distances = result_distances[batch_position]

            candidates: list[dict[str, Any]] = []

            for rank, (
                chroma_id,
                document,
                metadata,
                distance,
            ) in enumerate(
                zip(
                    ids,
                    documents,
                    metadatas,
                    distances,
                ),
                start=1,
            ):
                candidate = build_candidate(
                    rank=rank,
                    chroma_id=chroma_id,
                    distance=distance,
                    chroma_document=document,
                    chroma_metadata=metadata,
                    records_by_index=records_by_index,
                )

                candidates.append(candidate)

            retrieval_record = {
                **user_function,
                "retrieval": {
                    "embedding_model": EMBEDDING_MODEL,
                    "collection": COLLECTION_NAME,
                    "top_k": TOP_K,
                    "candidate_count": len(candidates),
                },
                "retrieved_candidates": candidates,
            }

            retrieval_results.append(retrieval_record)

            print_retrieval_summary(retrieval_record)

    return retrieval_results, skipped_methods


# Console output
def print_retrieval_summary(
    retrieval_record: dict[str, Any],
) -> None:
    """Print a readable summary without displaying full code."""

    print("\n" + "=" * 70)
    print(
        f"Function: "
        f"{retrieval_record['function_name']}"
    )
    print(f"Method ID: {retrieval_record['method_id']}")
    print(
        f"Location: {retrieval_record['file_path']}:"
        f"{retrieval_record['start_line']}-"
        f"{retrieval_record['end_line']}"
    )
    print(
        "Description: "
        f"{retrieval_record['functional_description']}"
    )
    print("\nTop retrieved candidates")
    print("-" * 70)

    for candidate in retrieval_record[
        "retrieved_candidates"
    ]:
        record = candidate["knowledge_record"]

        print(
            f"Rank {candidate['rank']} | "
            f"Similarity: {candidate['similarity']:.4f} | "
            f"ID: {candidate['source_index']} | "
            f"CWE: {record.get('cwe')} | "
            f"Subcategory: {record.get('subcategory')}"
        )

        print(
            "  "
            + str(
                record.get("functional_description", "")
            )
        )


# Output writers

def write_jsonl(
    records: list[dict[str, Any]],
    output_file: Path,
) -> None:
    """Write one retrieval result per user function."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_file.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_skipped_csv(
    skipped_methods: list[dict[str, str]],
    output_file: Path,
) -> None:
    """Write functions skipped because descriptions were empty."""

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "method_id",
        "file_path",
        "function_name",
        "start_line",
        "end_line",
        "reason",
    ]

    with output_file.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(skipped_methods)


# Verification
def verify_retrieval_results(
    retrieval_results: list[dict[str, Any]],
) -> None:
    """Check that each processed function received top-k results."""

    if not retrieval_results:
        raise RuntimeError("No retrieval results were generated.")

    invalid_results = []

    for record in retrieval_results:
        candidates = record.get("retrieved_candidates", [])

        if len(candidates) != TOP_K:
            invalid_results.append({
                "method_id": record.get("method_id"),
                "candidate_count": len(candidates),
            })

    if invalid_results:
        raise RuntimeError(
            "Some functions did not receive the expected "
            f"{TOP_K} candidates:\n"
            + json.dumps(
                invalid_results,
                ensure_ascii=False,
                indent=2,
            )
        )


# Main
def main() -> None:
    print("=" * 70)
    print("USER FUNCTION KNOWLEDGE RETRIEVAL")
    print("=" * 70)

    print(f"Project: {PROCESSED_PROJECT_NAME}")
    print(f"Top-k: {TOP_K}")

    require_file(
        METHOD_DATA_FILE,
        "method_data.csv",
    )

    require_file(
        KNOWLEDGE_BASE_FILE,
        "Knowledge-base JSONL file",
    )

    require_directory(
        CHROMA_DIRECTORY,
        "Chroma database directory",
    )

    print("\nInput files")
    print("-" * 70)
    print(f"Methods: {METHOD_DATA_FILE.resolve()}")
    print(
        f"Knowledge base: "
        f"{KNOWLEDGE_BASE_FILE.resolve()}"
    )

    methods = read_method_data(METHOD_DATA_FILE)

    records_by_index = read_knowledge_base(
        KNOWLEDGE_BASE_FILE
    )

    print(f"\nFunctions loaded: {len(methods)}")
    print(
        f"Full knowledge records loaded: "
        f"{len(records_by_index)}"
    )

    collection = open_chroma_collection()
    model = load_embedding_model()

    retrieval_results, skipped_methods = retrieve_for_methods(
        methods=methods,
        model=model,
        collection=collection,
        records_by_index=records_by_index,
    )

    verify_retrieval_results(retrieval_results)

    write_jsonl(
        retrieval_results,
        OUTPUT_FILE,
    )

    write_skipped_csv(
        skipped_methods,
        SKIPPED_FUNCTIONS_FILE,
    )

    print("\n" + "=" * 70)
    print("RETRIEVAL COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Functions in CSV: {len(methods)}")
    print(
        f"Functions retrieved: "
        f"{len(retrieval_results)}"
    )
    print(
        f"Functions skipped: "
        f"{len(skipped_methods)}"
    )
    print(f"Candidates per function: {TOP_K}")
    print(f"Results: {OUTPUT_FILE.resolve()}")
    print(
        f"Skipped functions: "
        f"{SKIPPED_FUNCTIONS_FILE.resolve()}"
    )


if __name__ == "__main__":
    main()
