import json
from pathlib import Path
from typing import Any


# ============================================================
# CONFIGURATION
# ============================================================

MINERU_DIR = Path(
    r"D:\Langchain\pdf_rag_q&a\mineru_output\attention\hybrid_auto"
)

INPUT_FILE = MINERU_DIR / "chunks.json"

OUTPUT_FILE = MINERU_DIR / "index_documents.json"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks(file_path: Path) -> list[dict[str, Any]]:
    """
    Load chunks created by chunker.py.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Chunks file not found:\n{file_path}"
        )

    with open(file_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError(
            "Expected chunks.json to contain a JSON list."
        )

    return chunks


# ============================================================
# HELPERS
# ============================================================

def normalize_content_types(metadata: dict) -> list[str]:
    """
    Return content_types as a clean list.

    Example:
        ["text"]
        ["text", "equation"]
        ["table"]
        ["figure"]
    """

    content_types = metadata.get("content_types", [])

    if isinstance(content_types, str):
        content_types = [content_types]

    if not isinstance(content_types, list):
        return []

    return [
        str(content_type).strip().lower()
        for content_type in content_types
        if str(content_type).strip()
    ]


def normalize_section_path(metadata: dict) -> list[str]:
    """
    Return section_path as a clean list.
    """

    section_path = metadata.get("section_path", [])

    if isinstance(section_path, str):
        section_path = [section_path]

    if not isinstance(section_path, list):
        return []

    return [
        str(section).strip()
        for section in section_path
        if str(section).strip()
    ]


def is_reference_section(section_path: list[str]) -> bool:
    """
    Detect whether a chunk belongs to the References section.
    """

    return any(
        section.strip().lower() == "references"
        for section in section_path
    )


def is_structured_content(content_types: list[str]) -> bool:
    """
    Structured content should normally remain searchable even
    when MinerU assigns it to an imperfect section.

    Examples:
        table
        figure
        equation
    """

    structured_types = {
        "table",
        "figure",
        "equation",
    }

    return any(
        content_type in structured_types
        for content_type in content_types
    )


def should_index_chunk(chunk: dict) -> tuple[bool, str]:
    """
    Decide whether a chunk should enter the retrieval index.

    Current policy:

    1. Empty chunks are excluded.

    2. Pure reference-list text is excluded because reference
       entries usually add noise to semantic retrieval.

    3. Structured content such as figures, tables and equations
       is retained even if MinerU assigned it to References.

       This protects against layout/section-classification errors.
    """

    page_content = str(
        chunk.get("page_content", "")
    ).strip()

    metadata = chunk.get("metadata", {})

    if not page_content:
        return False, "empty_content"

    if not isinstance(metadata, dict):
        return False, "invalid_metadata"

    content_types = normalize_content_types(metadata)
    section_path = normalize_section_path(metadata)

    if is_reference_section(section_path):

        if not is_structured_content(content_types):
            return False, "reference_text"

    return True, "included"


# ============================================================
# RETRIEVAL TYPE
# ============================================================

def get_retrieval_type(content_types: list[str]) -> str:
    """
    Give each index document a simple retrieval category.

    Examples:

        ["text"]
            -> text

        ["table"]
            -> table

        ["text", "equation"]
            -> mixed

        ["text", "figure"]
            -> mixed
    """

    unique_types = set(content_types)

    if unique_types == {"text"}:
        return "text"

    if unique_types == {"table"}:
        return "table"

    if unique_types == {"figure"}:
        return "figure"

    if unique_types == {"equation"}:
        return "equation"

    return "mixed"


# ============================================================
# PREPARE INDEX DOCUMENTS
# ============================================================

def prepare_index_documents(
    chunks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Convert validated chunks into documents ready for embeddings.

    Returns:

        index_documents
        excluded_documents
    """

    index_documents = []
    excluded_documents = []

    for source_index, chunk in enumerate(chunks):

        should_index, reason = should_index_chunk(chunk)

        metadata = chunk.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        metadata = metadata.copy()

        content_types = normalize_content_types(metadata)
        section_path = normalize_section_path(metadata)

        if not should_index:

            excluded_documents.append(
                {
                    "source_index": source_index,
                    "chunk_id": metadata.get("chunk_id"),
                    "reason": reason,
                    "section_path": section_path,
                    "content_types": content_types,
                }
            )

            continue

        # ----------------------------------------------------
        # Add retrieval-specific metadata
        # ----------------------------------------------------

        metadata["index_id"] = len(index_documents)

        metadata["source_chunk_index"] = source_index

        metadata["retrieval_type"] = get_retrieval_type(
            content_types
        )

        metadata["is_reference_section"] = (
            is_reference_section(section_path)
        )

        metadata["has_image"] = bool(
            metadata.get("image_paths")
        )

        # ----------------------------------------------------
        # Keep original chunk text unchanged
        # ----------------------------------------------------

        index_document = {
            "page_content": chunk["page_content"],
            "metadata": metadata,
        }

        index_documents.append(index_document)

    return index_documents, excluded_documents


# ============================================================
# VALIDATION
# ============================================================

def validate_index_documents(
    documents: list[dict[str, Any]]
) -> None:
    """
    Basic validation before embedding.
    """

    empty_documents = 0
    metadata_errors = 0
    duplicate_documents = 0

    seen_content = set()

    retrieval_types = {}

    for document in documents:

        page_content = str(
            document.get("page_content", "")
        ).strip()

        metadata = document.get("metadata", {})

        # ----------------------------------------------------
        # Empty content
        # ----------------------------------------------------

        if not page_content:
            empty_documents += 1

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        required_metadata = [
            "index_id",
            "chunk_id",
            "section_path",
            "content_types",
            "retrieval_type",
        ]

        for key in required_metadata:

            if key not in metadata:
                metadata_errors += 1
                break

        # ----------------------------------------------------
        # Duplicate content
        # ----------------------------------------------------

        if page_content in seen_content:
            duplicate_documents += 1
        else:
            seen_content.add(page_content)

        # ----------------------------------------------------
        # Retrieval type summary
        # ----------------------------------------------------

        retrieval_type = metadata.get(
            "retrieval_type",
            "unknown"
        )

        retrieval_types[retrieval_type] = (
            retrieval_types.get(retrieval_type, 0) + 1
        )

    print("\nINDEX VALIDATION")
    print("=" * 60)

    print(f"Total index documents : {len(documents)}")
    print(f"Empty documents       : {empty_documents}")
    print(f"Metadata errors       : {metadata_errors}")
    print(f"Duplicate documents   : {duplicate_documents}")

    print("\nRetrieval types:")

    for retrieval_type, count in sorted(
        retrieval_types.items()
    ):
        print(
            f"  {retrieval_type:<10} : {count}"
        )

    print("=" * 60)

    if (
        empty_documents == 0
        and metadata_errors == 0
        and duplicate_documents == 0
    ):
        print("INDEX VALIDATION STATUS : PASS")
    else:
        print("INDEX VALIDATION STATUS : REVIEW")


# ============================================================
# SAVE
# ============================================================

def save_index_documents(
    documents: list[dict[str, Any]],
    output_file: Path,
) -> None:

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            documents,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\nSaved index documents to:\n{output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\nLoading chunks...")

    chunks = load_chunks(INPUT_FILE)

    print(
        f"Loaded {len(chunks)} chunks"
    )

    # --------------------------------------------------------
    # Prepare documents
    # --------------------------------------------------------

    index_documents, excluded_documents = (
        prepare_index_documents(chunks)
    )

    print(
        f"Prepared {len(index_documents)} index documents"
    )

    print(
        f"Excluded {len(excluded_documents)} chunks"
    )

    # --------------------------------------------------------
    # Show excluded chunks
    # --------------------------------------------------------

    if excluded_documents:

        print("\nEXCLUDED CHUNKS")
        print("=" * 60)

        for item in excluded_documents:

            print(
                f"Chunk {item['chunk_id']} | "
                f"Reason: {item['reason']} | "
                f"Section: {item['section_path']} | "
                f"Types: {item['content_types']}"
            )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_index_documents(
        index_documents
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_index_documents(
        index_documents,
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()