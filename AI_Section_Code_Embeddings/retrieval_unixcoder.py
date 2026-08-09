from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma

from embeddings_unixcoder import UniXcoderEmbeddings


# ============================================================
# Project configuration
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent


# ------------------------------------------------------------
# Benchmark / processed project
# ------------------------------------------------------------

PROCESSED_PROJECT_NAME = "testcode"


# ------------------------------------------------------------
# Preprocessing output
# ------------------------------------------------------------

# Change ONLY this filename if your preprocessing CSV has a
# different name.
METHODS_FILE = Path(
    r"C:\Users\ASUST\Desktop\RAG Based Security Auditor\processed\testcode\method_data_owasp_post_only.csv"
)


# ------------------------------------------------------------
# Knowledge base
# ------------------------------------------------------------

KNOWLEDGE_BASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "final-knowledge-base.jsonl"
)


# ------------------------------------------------------------
# UniXcoder Chroma database
# ------------------------------------------------------------

UNIXCODER_CHROMA_DIR = (
    PROJECT_ROOT
    / "chroma_db_unixcoder"
)

UNIXCODER_COLLECTION_NAME = (
    "securecode_unixcoder_vulnerable_code"
)

UNIXCODER_MODEL_NAME = (
    "microsoft/unixcoder-base"
)


# ------------------------------------------------------------
# Retrieval output
# ------------------------------------------------------------

RETRIEVAL_OUTPUT_FILE = Path(
    r"C:\Users\ASUST\Desktop\RAG Based Security Auditor\processed\testcode\retrieval_results_unixcoder_2.jsonl"
)

RETRIEVAL_ERRORS_FILE = Path(
    r"C:\Users\ASUST\Desktop\RAG Based Security Auditor\processed\testcode\retrieval_errors_unixcoder_2.csv"
)

RETRIEVAL_SUMMARY_FILE = Path(
    r"C:\Users\ASUST\Desktop\RAG Based Security Auditor\processed\testcode\retrieval_summary_unixcoder_2.json"
)
# ------------------------------------------------------------
# Retrieval settings
# ------------------------------------------------------------

TOP_K = 3


# ============================================================
# File helpers
# ============================================================

def require_file(
    file_path: Path,
    description: str,
) -> None:

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


def write_jsonl(
    records: list[dict[str, Any]],
    file_path: Path,
) -> None:

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
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


def write_json(
    record: dict[str, Any],
    file_path: Path,
) -> None:

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Read preprocessing CSV
# ============================================================

def load_method_rows(
    csv_path: Path,
) -> list[dict[str, str]]:

    with csv_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        required_columns = {
            "method_id",
            "file_path",
            "language",
            "name",
            "start_line",
            "end_line",
            "source_code",
        }

        headers = set(
            reader.fieldnames or []
        )

        missing_columns = (
            required_columns - headers
        )

        if missing_columns:
            raise ValueError(
                "Method CSV is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        rows = list(reader)

    if not rows:
        raise ValueError(
            f"No method rows were found in:\n"
            f"{csv_path.resolve()}"
        )

    return rows


# ============================================================
# Read complete knowledge base
# ============================================================

def load_knowledge_base(
    file_path: Path,
) -> list[dict[str, Any]]:

    records: list[dict[str, Any]] = []

    with file_path.open(
        "r",
        encoding="utf-8-sig",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number} of "
                    f"{file_path.name}: {error}"
                ) from error

            if not isinstance(
                record,
                dict,
            ):
                raise ValueError(
                    f"Line {line_number} of "
                    f"{file_path.name} is not "
                    f"a JSON object."
                )

            records.append(record)

    if not records:
        raise ValueError(
            "The knowledge base is empty."
        )

    return records


# ============================================================
# Optional helper
# ============================================================

def safe_int(
    value: Any,
    default: int = 0,
) -> int:

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================
# Load UniXcoder + Chroma
# ============================================================

def load_vectorstore() -> Chroma:

    print(
        "[INFO] Loading UniXcoder model..."
    )

    embedding_function = (
        UniXcoderEmbeddings(
            model_name=UNIXCODER_MODEL_NAME,
        )
    )

    print(
        "[INFO] Loading UniXcoder Chroma collection..."
    )

    vectorstore = Chroma(
        collection_name=(
            UNIXCODER_COLLECTION_NAME
        ),
        embedding_function=(
            embedding_function
        ),
        persist_directory=str(
            UNIXCODER_CHROMA_DIR
        ),
        collection_metadata={
            "hnsw:space": "cosine",
        },
    )

    return vectorstore


# ============================================================
# Retrieve for one function
# ============================================================

def retrieve_for_function(
    row: dict[str, str],
    vectorstore: Chroma,
    knowledge_records: list[dict[str, Any]],
) -> dict[str, Any]:

    source_code = str(
        row.get(
            "source_code",
            "",
        )
    ).strip()

    if not source_code:
        raise ValueError(
            "source_code is empty."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Query UniXcoder using the ACTUAL SOURCE CODE.
    # Do not use the LLM functional description here.
    # --------------------------------------------------------

    results = (
        vectorstore.similarity_search_with_score(
            source_code,
            k=TOP_K,
        )
    )

    retrieved_candidates: list[
        dict[str, Any]
    ] = []

    for rank, (
        document,
        distance,
    ) in enumerate(
        results,
        start=1,
    ):

        metadata = (
            document.metadata or {}
        )

        source_index = safe_int(
            metadata.get(
                "source_index"
            ),
            default=-1,
        )

        if (
            source_index < 0
            or source_index
            >= len(knowledge_records)
        ):
            raise ValueError(
                "Invalid source_index returned "
                f"from Chroma: {source_index}"
            )

        knowledge_record = (
            knowledge_records[
                source_index
            ]
        )

        # Chroma uses cosine distance because the collection
        # was created with hnsw:space="cosine".
        #
        # distance:
        #   smaller = more similar
        #
        # similarity:
        #   larger = more similar
        #
        # cosine similarity = 1 - cosine distance
        similarity = (
            1.0 - float(distance)
        )

        retrieved_candidates.append(
            {
                "rank": rank,

                "source_index": (
                    source_index
                ),

                "similarity": round(
                    similarity,
                    6,
                ),

                "distance": round(
                    float(distance),
                    6,
                ),

                # Complete KB record is hydrated here so
                # vulnerability_reasoning.py can use it.
                "knowledge_record": (
                    knowledge_record
                ),
            }
        )

    # --------------------------------------------------------
    # Preserve all fields needed by vulnerability reasoning
    # --------------------------------------------------------

    functional_description = str(
        row.get(
            "functional_description",
            row.get(
                "description",
                "",
            ),
        )
        or ""
    ).strip()

    return {
        "method_id": row.get(
            "method_id",
            ""
        ),

        "unit_type": row.get(
            "unit_type",
            "function",
        ),

        "file_path": row.get(
            "file_path",
            "",
        ),

        "language": row.get(
            "language",
            "",
        ),

        "class_name": row.get(
            "class_name",
            "",
        ),

        "function_name": row.get(
            "name",
            "",
        ),

        "start_line": safe_int(
            row.get(
                "start_line"
            )
        ),

        "end_line": safe_int(
            row.get(
                "end_line"
            )
        ),

        "source_code": source_code,

        # Preserve this for the later reasoning phase.
        # It is NOT used for UniXcoder retrieval.
        "functional_description": (
            functional_description
        ),

        "references": row.get(
            "references",
            "",
        ),

        "retrieval_model": (
            UNIXCODER_MODEL_NAME
        ),

        "retrieval_representation": (
            "source_code"
        ),

        "top_k": TOP_K,

        "retrieved_candidates": (
            retrieved_candidates
        ),
    }


# ============================================================
# Main retrieval pipeline
# ============================================================

def main() -> None:

    print("=" * 70)
    print("UNIXCODER FUNCTION-LEVEL RETRIEVAL")
    print("=" * 70)

    print(
        f"Project: "
        f"{PROCESSED_PROJECT_NAME}"
    )

    print(
        f"Model: "
        f"{UNIXCODER_MODEL_NAME}"
    )

    print(
        f"Top-K: "
        f"{TOP_K}"
    )

    print(
        f"Method CSV: "
        f"{METHODS_FILE.resolve()}"
    )

    print(
        f"Knowledge base: "
        f"{KNOWLEDGE_BASE_FILE.resolve()}"
    )

    print(
        f"Chroma directory: "
        f"{UNIXCODER_CHROMA_DIR.resolve()}"
    )

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    require_file(
        METHODS_FILE,
        "Preprocessed method CSV",
    )

    require_file(
        KNOWLEDGE_BASE_FILE,
        "final-knowledge-base.jsonl",
    )

    if not UNIXCODER_CHROMA_DIR.exists():
        raise FileNotFoundError(
            "UniXcoder Chroma database was not found:\n"
            f"{UNIXCODER_CHROMA_DIR.resolve()}\n"
            "Run UniXcoder ingestion first."
        )

    # --------------------------------------------------------
    # Load input data
    # --------------------------------------------------------

    method_rows = load_method_rows(
        METHODS_FILE
    )

    knowledge_records = (
        load_knowledge_base(
            KNOWLEDGE_BASE_FILE
        )
    )

    print()
    print(
        f"[INFO] Functions loaded: "
        f"{len(method_rows)}"
    )

    print(
        f"[INFO] Knowledge-base records loaded: "
        f"{len(knowledge_records)}"
    )

    # --------------------------------------------------------
    # Load UniXcoder exactly ONCE
    # --------------------------------------------------------

    vectorstore = load_vectorstore()

    # --------------------------------------------------------
    # Verify Chroma collection
    # --------------------------------------------------------

    collection_data = vectorstore.get(
        include=[],
    )

    stored_ids = collection_data.get(
        "ids",
        [],
    )

    print(
        f"[INFO] Chroma vectors found: "
        f"{len(stored_ids)}"
    )

    if not stored_ids:
        raise ValueError(
            "The UniXcoder Chroma collection "
            "contains no vectors."
        )

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieval_results: list[
        dict[str, Any]
    ] = []

    errors: list[
        dict[str, Any]
    ] = []

    total = len(method_rows)

    print()
    print(
        "[INFO] Beginning code-to-code retrieval..."
    )

    for position, row in enumerate(
        method_rows,
        start=1,
    ):

        method_id = str(
            row.get(
                "method_id",
                "",
            )
        )

        function_name = str(
            row.get(
                "name",
                "",
            )
        )

        print(
            f"[{position}/{total}] "
            f"Retrieving: {function_name}"
        )

        try:

            result = retrieve_for_function(
                row=row,
                vectorstore=vectorstore,
                knowledge_records=(
                    knowledge_records
                ),
            )

            retrieval_results.append(
                result
            )

            candidates = result[
                "retrieved_candidates"
            ]

            if candidates:

                top_result = candidates[0]

                top_record = top_result[
                    "knowledge_record"
                ]

                top_cwe = (
                    top_record.get("cwe")
                    or top_record.get(
                        "cwe_id"
                    )
                    or top_record.get(
                        "CWE"
                    )
                    or "-"
                )

                print(
                    f"  Top-1: "
                    f"source_index="
                    f"{top_result['source_index']} | "
                    f"CWE={top_cwe} | "
                    f"similarity="
                    f"{top_result['similarity']:.4f}"
                )

        except Exception as error:

            errors.append(
                {
                    "method_id": method_id,
                    "function_name": (
                        function_name
                    ),
                    "file_path": row.get(
                        "file_path",
                        "",
                    ),
                    "error": str(error),
                }
            )

            print(
                f"  ERROR: {error}"
            )

    # --------------------------------------------------------
    # Save retrieval results
    # --------------------------------------------------------

    write_jsonl(
        retrieval_results,
        RETRIEVAL_OUTPUT_FILE,
    )

    # --------------------------------------------------------
    # Save errors
    # --------------------------------------------------------

    RETRIEVAL_ERRORS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RETRIEVAL_ERRORS_FILE.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:

        fieldnames = [
            "method_id",
            "function_name",
            "file_path",
            "error",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            errors
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "project": (
            PROCESSED_PROJECT_NAME
        ),

        "embedding_model": (
            UNIXCODER_MODEL_NAME
        ),

        "retrieval_representation": (
            "source_code"
        ),

        "collection_name": (
            UNIXCODER_COLLECTION_NAME
        ),

        "top_k": TOP_K,

        "functions_supplied": total,

        "successful_retrievals": len(
            retrieval_results
        ),

        "retrieval_failures": len(
            errors
        ),

        "knowledge_base_records": len(
            knowledge_records
        ),

        "stored_chroma_vectors": len(
            stored_ids
        ),
    }

    write_json(
        summary,
        RETRIEVAL_SUMMARY_FILE,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("UNIXCODER RETRIEVAL COMPLETED")
    print("=" * 70)

    print(
        f"Functions supplied: "
        f"{total}"
    )

    print(
        f"Successful retrievals: "
        f"{len(retrieval_results)}"
    )

    print(
        f"Failures: "
        f"{len(errors)}"
    )

    print(
        f"Top-K: "
        f"{TOP_K}"
    )

    print(
        f"Results: "
        f"{RETRIEVAL_OUTPUT_FILE.resolve()}"
    )

    print(
        f"Errors: "
        f"{RETRIEVAL_ERRORS_FILE.resolve()}"
    )

    print(
        f"Summary: "
        f"{RETRIEVAL_SUMMARY_FILE.resolve()}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()