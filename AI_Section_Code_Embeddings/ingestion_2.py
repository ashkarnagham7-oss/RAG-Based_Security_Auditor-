from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
from typing import Any
from langchain_chroma import Chroma
from langchain_core.documents import Document
from embeddings_unixcoder import UniXcoderEmbeddings

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIRECTORY.parent

KNOWLEDGE_BASE_FILE = (
    PROJECT_ROOT
    / "data"
    / "final-knowledge-base.jsonl"
)

# UniXcoder Chroma database
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


# Ingestion configuration


CHROMA_BATCH_SIZE = 64


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


def safe_metadata_value(
    value: Any,
) -> str | int | float | bool:
    """
    Chroma metadata should contain simple scalar values.

    None, lists, dictionaries, and other complex values are
    converted into safe representations.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        (list, dict),
    ):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    return str(value)


def first_present(
    record: dict[str, Any],
    field_names: list[str],
    default: Any = None,
) -> Any:
    """
    Return the first non-empty field found.
    """

    for field_name in field_names:
        value = record.get(field_name)

        if value is not None and value != "":
            return value

    return default



def load_knowledge_base(
    file_path: Path,
) -> list[dict[str, Any]]:
    """
    Read the JSONL knowledge base.

    source_index is intentionally based on the original JSONL
    line/record position so retrieval can later hydrate the
    complete knowledge record.
    """

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

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} of "
                    f"{file_path.name} is not "
                    f"a JSON object."
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No knowledge records were found in:\n"
            f"{file_path.resolve()}"
        )

    return records



# Convert KB records into Chroma documents

def build_documents(
    knowledge_records: list[dict[str, Any]],
) -> tuple[
    list[Document],
    list[str],
]:
    """
    Build Chroma documents.

    IMPORTANT:
    UniXcoder embeds ONLY vulnerable_code.

    Other knowledge fields are stored as lightweight metadata
    only. The complete knowledge record remains in the JSONL
    file and can be hydrated during retrieval using source_index.
    """

    documents: list[Document] = []
    document_ids: list[str] = []

    skipped_missing_vulnerable_code = 0

    for source_index, record in enumerate(
        knowledge_records
    ):

        vulnerable_code = first_present(
            record,
            [
                "vulnerable_code",
                "unsafe_code",
                "bad_code",
                "code_before",
            ],
        )

        if vulnerable_code is None:
            skipped_missing_vulnerable_code += 1
            continue

        vulnerable_code = str(
            vulnerable_code
        ).strip()

        if not vulnerable_code:
            skipped_missing_vulnerable_code += 1
            continue

        # Metadata
        cwe = first_present(
            record,
            [
                "cwe",
                "cwe_id",
                "CWE",
            ],
            "",
        )

        language = first_present(
            record,
            [
                "language",
                "lang",
            ],
            "",
        )

        category = first_present(
            record,
            [
                "category",
                "vulnerability_category",
                "owasp_category",
            ],
            "",
        )

        subcategory = first_present(
            record,
            [
                "subcategory",
                "vulnerability_subcategory",
            ],
            "",
        )

        metadata = {
            "source_index": source_index,
            "cwe": safe_metadata_value(cwe),
            "language": safe_metadata_value(
                language
            ),
            "category": safe_metadata_value(
                category
            ),
            "subcategory": safe_metadata_value(
                subcategory
            ),
        }

       
        # Chroma document

        # page_content is exactly the code representation that
        # UniXcoder will embed.
        document = Document(
            page_content=vulnerable_code,
            metadata=metadata,
        )

        documents.append(document)

    
        document_ids.append(
            f"kb_{source_index}"
        )

    print(
        f"[INFO] Knowledge-base records loaded: "
        f"{len(knowledge_records)}"
    )

    print(
        f"[INFO] Vulnerable-code documents prepared: "
        f"{len(documents)}"
    )

    print(
        f"[INFO] Records skipped because vulnerable_code "
        f"was missing/empty: "
        f"{skipped_missing_vulnerable_code}"
    )

    if not documents:
        raise ValueError(
            "No vulnerable-code documents were available "
            "for UniXcoder ingestion."
        )

    return documents, document_ids


# Chroma ingestion


def build_unixcoder_chroma() -> Chroma:
    """
    Rebuild the UniXcoder Chroma database from scratch.

    Embedding representation:
        vulnerable_code -> UniXcoder -> Chroma

    The complete knowledge records remain in
    final-knowledge-base.jsonl.
    """

    print("=" * 70)
    print("UNIXCODER KNOWLEDGE-BASE INGESTION")
    print("=" * 70)

    print(
        f"Knowledge base: "
        f"{KNOWLEDGE_BASE_FILE.resolve()}"
    )

    print(
        f"UniXcoder model: "
        f"{UNIXCODER_MODEL_NAME}"
    )

    print(
        f"Chroma directory: "
        f"{UNIXCODER_CHROMA_DIR.resolve()}"
    )

    print(
        f"Collection: "
        f"{UNIXCODER_COLLECTION_NAME}"
    )

    # Validate knowledge base

    require_file(
        KNOWLEDGE_BASE_FILE,
        "final-knowledge-base.jsonl",
    )

    # Load KB
    knowledge_records = load_knowledge_base(
        KNOWLEDGE_BASE_FILE
    )

    documents, document_ids = build_documents(
        knowledge_records
    )

    # Start with a completely clean UniXcoder DB
    if UNIXCODER_CHROMA_DIR.exists():

        print(
            "[INFO] Existing UniXcoder Chroma directory "
            "found."
        )

        print(
            "[INFO] Removing old UniXcoder database..."
        )

        shutil.rmtree(
            UNIXCODER_CHROMA_DIR
        )

    UNIXCODER_CHROMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Load UniXcoder exactly once

    print()
    print(
        "[INFO] Loading UniXcoder embedding model..."
    )

    embedding_function = (
        UniXcoderEmbeddings(
            model_name=UNIXCODER_MODEL_NAME,
        )
    )

    # Create Chroma collection

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

    # Add documents in manageable batches

    total_documents = len(documents)

    print()
    print(
        f"[INFO] Beginning ingestion of "
        f"{total_documents} vulnerable-code examples..."
    )

    for start_index in range(
        0,
        total_documents,
        CHROMA_BATCH_SIZE,
    ):

        end_index = min(
            start_index + CHROMA_BATCH_SIZE,
            total_documents,
        )

        batch_documents = documents[
            start_index:end_index
        ]

        batch_ids = document_ids[
            start_index:end_index
        ]

        vectorstore.add_documents(
            documents=batch_documents,
            ids=batch_ids,
        )

        print(
            f"[INFO] Embedded and stored "
            f"{end_index}/{total_documents}"
        )

    # Verify collection

    stored = vectorstore.get(
        include=[],
    )

    stored_ids = stored.get(
        "ids",
        [],
    )

    stored_count = len(
        stored_ids
    )

    if stored_count != total_documents:
        raise RuntimeError(
            "Chroma verification failed. "
            f"Expected {total_documents} records, "
            f"but found {stored_count}."
        )

    # Final information
    print()
    print("=" * 70)
    print("UNIXCODER INGESTION COMPLETED")
    print("=" * 70)

    print(
        f"KB records: "
        f"{len(knowledge_records)}"
    )

    print(
        f"Embedded vulnerable-code examples: "
        f"{total_documents}"
    )

    print(
        f"Stored Chroma vectors: "
        f"{stored_count}"
    )

    print(
        f"Model: "
        f"{UNIXCODER_MODEL_NAME}"
    )

    print(
        f"Chroma directory: "
        f"{UNIXCODER_CHROMA_DIR.resolve()}"
    )

    print(
        f"Collection: "
        f"{UNIXCODER_COLLECTION_NAME}"
    )

    return vectorstore

if __name__ == "__main__":
    build_unixcoder_chroma()