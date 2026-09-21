from documents import (
    load_elements,
    get_heading_level,
)

from langchain_core.documents import Document
import json
from pathlib import Path


MINERU_DIR = Path(
    r"D:\Langchain\pdf_rag_q&a\mineru_output\attention\hybrid_auto"
)

CHUNKS_OUTPUT_FILE = (
    MINERU_DIR / "chunks.json"
)

# ============================================================
# CONFIGURATION
# ============================================================

MAX_CHUNK_CHARS = 3000


# ============================================================
# SECTION
# ============================================================

def update_section_path(
    current_path,
    heading
):

    level = get_heading_level(
        heading
    )

    current_path = (
        current_path[:level - 1]
    )

    current_path.append(
        heading
    )

    return current_path


# ============================================================
# FIND CONTENT START
# ============================================================

def find_content_start(elements):

    for index, element in enumerate(elements):

        if element.get(
            "content_type"
        ) == "heading":

            heading = element.get(
                "text",
                ""
            ).strip()

            if heading:

                return index

    return 0


# ============================================================
# FORMAT ELEMENT
# ============================================================

def format_element(element):

    content_type = element.get(
        "content_type"
    )

    text = element.get(
        "text",
        ""
    ).strip()

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    if content_type == "text":

        return text

    # --------------------------------------------------------
    # EQUATION
    # --------------------------------------------------------

    if content_type == "equation":

        return (
            "[Equation]\n"
            f"{text}"
        )

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    if content_type == "figure":

        caption = element.get(
            "caption",
            ""
        ).strip()

        if caption:

            return (
                "[Figure]\n"
                f"{caption}"
            )

        return (
            "[Figure]\n"
            f"{text}"
        )

    # --------------------------------------------------------
    # TABLE
    # --------------------------------------------------------

    if content_type == "table":

        caption = element.get(
            "caption",
            ""
        ).strip()

        html = element.get(
            "html",
            ""
        ).strip()

        return (
            "[Table]\n"
            f"{caption}\n\n"
            f"{html}"
        ).strip()

    return ""


# ============================================================
# CREATE CHUNK
# ============================================================

def create_chunk(
    elements,
    section_path,
    chunk_id
):

    if not elements:

        return None

    content_parts = []

    content_types = []

    pages = []

    element_ids = []

    image_paths = []

    for element in elements:

        formatted = format_element(
            element
        )

        if not formatted:

            continue

        content_parts.append(
            formatted
        )

        content_type = element.get(
            "content_type"
        )

        content_types.append(
            content_type
        )

        page = element.get(
            "page"
        )

        if page is not None:

            pages.append(page)

        element_id = element.get(
            "element_id"
        )

        if element_id is not None:

            element_ids.append(
                element_id
            )

        image_path = element.get(
            "image_path"
        )

        if image_path:

            image_paths.append(
                image_path
            )

    if not content_parts:

        return None

    page_content = (
        "\n\n".join(
            content_parts
        )
    ).strip()

    metadata = {

        "source": "attention.pdf",

        "chunk_id": chunk_id,

        "section_path": (
            section_path.copy()
        ),

        "content_types": list(
            dict.fromkeys(
                content_types
            )
        ),

        "pages": sorted(
            set(pages)
        ),

        "element_ids": element_ids,

        "chunk_size": len(
            page_content
        ),
    }

    if image_paths:

        metadata["image_paths"] = (
            image_paths
        )

    return Document(
        page_content=page_content,
        metadata=metadata
    )


# ============================================================
# FLUSH
# ============================================================

def flush_chunk(
    chunks,
    buffer,
    section_path,
    chunk_id
):

    if not buffer:

        return chunk_id

    chunk = create_chunk(
        buffer,
        section_path,
        chunk_id
    )

    if chunk:

        chunks.append(
            chunk
        )

        return chunk_id + 1

    return chunk_id


# ============================================================
# STRUCTURE-AWARE CHUNKING
# ============================================================

# ============================================================
# STRUCTURE-AWARE CHUNKING
# ============================================================

def create_chunks(elements):
    """
    Create structure-aware LangChain chunks.

    Policy:
    - Normal text is grouped up to MAX_CHUNK_CHARS.
    - Tables remain atomic.
    - Equations remain attached to nearby text when possible.
    - Figures remain attached to nearby text when possible.
    - Very small standalone text chunks are merged with
      the next text chunk when safely possible.
    """

    chunks = []

    current_section_path = []

    buffer = []

    chunk_id = 0

    content_start = find_content_start(elements)

    elements = elements[content_start:]

    # --------------------------------------------------------
    # Helper: flush current buffer
    # --------------------------------------------------------

    def flush_current_buffer():
        nonlocal buffer
        nonlocal chunk_id

        if not buffer:
            return

        chunk = create_chunk(
            buffer,
            current_section_path,
            chunk_id
        )

        if chunk:
            chunks.append(chunk)
            chunk_id += 1

        buffer = []

    # --------------------------------------------------------
    # Process elements
    # --------------------------------------------------------

    for element in elements:

        content_type = element.get(
            "content_type"
        )

        text = element.get(
            "text",
            ""
        ).strip()

        # ====================================================
        # HEADING
        # ====================================================

        if content_type == "heading":

            # Finish previous chunk.
            flush_current_buffer()

            # Update section hierarchy.
            if text:

                current_section_path = (
                    update_section_path(
                        current_section_path,
                        text
                    )
                )

            continue

        # ====================================================
        # EMPTY ELEMENT
        # ====================================================

        if (
            not text
            and content_type != "table"
        ):
            continue

        # ====================================================
        # TABLE
        # ====================================================

        if content_type == "table":

            # Tables are atomic.
            flush_current_buffer()

            chunk = create_chunk(
                [element],
                current_section_path,
                chunk_id
            )

            if chunk:

                chunks.append(chunk)

                chunk_id += 1

            continue

        # ====================================================
        # EQUATION / FIGURE
        # ====================================================

        if content_type in {
            "equation",
            "figure"
        }:

            candidate = (
                buffer + [element]
            )

            candidate_text = (
                "\n\n".join(
                    format_element(e)
                    for e in candidate
                )
            )

            # Attach to current chunk if
            # it remains within the limit.
            if (
                buffer
                and
                len(candidate_text)
                <= MAX_CHUNK_CHARS
            ):

                buffer.append(element)

            else:

                flush_current_buffer()

                buffer = [element]

            continue

        # ====================================================
        # NORMAL TEXT
        # ====================================================

        if content_type == "text":

            candidate = (
                buffer + [element]
            )

            candidate_text = (
                "\n\n".join(
                    format_element(e)
                    for e in candidate
                )
            )

            # Add if within target size.
            if (
                not buffer
                or
                len(candidate_text)
                <= MAX_CHUNK_CHARS
            ):

                buffer.append(element)

            else:

                flush_current_buffer()

                buffer = [element]

    # ========================================================
    # FINAL BUFFER
    # ========================================================

    flush_current_buffer()

    # ========================================================
    # MERGE VERY SMALL TEXT CHUNKS
    # ========================================================

    merged_chunks = []

    i = 0

    while i < len(chunks):

        current = chunks[i]

        current_size = len(
            current.page_content
        )

        current_types = current.metadata.get(
            "content_types",
            []
        )

        # ----------------------------------------------------
        # Only merge pure-text tiny chunks.
        # ----------------------------------------------------

        if (
            current_size < 200
            and current_types == ["text"]
            and i + 1 < len(chunks)
        ):

            next_chunk = chunks[i + 1]

            next_types = next_chunk.metadata.get(
                "content_types",
                []
            )

            # Only merge with another text chunk.
            if next_types == ["text"]:

                combined_content = (
                    current.page_content
                    + "\n\n"
                    + next_chunk.page_content
                )

                # Keep the metadata from the more
                # informative next chunk, but preserve
                # the current chunk's element IDs/pages.
                combined_metadata = dict(
                    next_chunk.metadata
                )

                combined_metadata["pages"] = sorted(
                    set(
                        current.metadata.get(
                            "pages",
                            []
                        )
                        +
                        next_chunk.metadata.get(
                            "pages",
                            []
                        )
                    )
                )

                combined_metadata["element_ids"] = (
                    current.metadata.get(
                        "element_ids",
                        []
                    )
                    +
                    next_chunk.metadata.get(
                        "element_ids",
                        []
                    )
                )

                combined_metadata["chunk_size"] = (
                    len(combined_content)
                )

                merged_chunks.append(
                    Document(
                        page_content=combined_content,
                        metadata=combined_metadata
                    )
                )

                i += 2

                continue

        # ----------------------------------------------------
        # Normal chunk
        # ----------------------------------------------------

        merged_chunks.append(current)

        i += 1

    # ========================================================
    # REASSIGN CHUNK IDS
    # ========================================================

    for index, chunk in enumerate(
        merged_chunks
    ):

        chunk.metadata["chunk_id"] = index

        chunk.metadata["chunk_size"] = (
            len(chunk.page_content)
        )

    return merged_chunks


# ============================================================
# PRINT
# ============================================================

def print_chunks(chunks):

    print(
        "\n" + "=" * 60
    )

    print(
        "GENERATED CHUNKS"
    )

    print(
        "=" * 60
    )

    for chunk in chunks:

        print(
            f"\n--- Chunk "
            f"{chunk.metadata['chunk_id']} ---"
        )

        print(
            "Pages:",
            chunk.metadata[
                "pages"
            ]
        )

        print(
            "Section:",
            chunk.metadata[
                "section_path"
            ]
        )

        print(
            "Content types:",
            chunk.metadata[
                "content_types"
            ]
        )

        print(
            "Size:",
            chunk.metadata[
                "chunk_size"
            ]
        )

        print(
            "Content:"
        )

        print(
            chunk.page_content[:700]
        )


# ============================================================
# SAVING CHUNKS
# ============================================================
def save_chunks(chunks):

    serialized = []

    for chunk in chunks:

        serialized.append({

            "page_content":
                chunk.page_content,

            "metadata":
                chunk.metadata,

        })

    with open(
        CHUNKS_OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            serialized,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved chunks to:\n"
        f"{CHUNKS_OUTPUT_FILE}"
    )



# ============================================================
# VALIDATING CHUNKS
# ============================================================
def validate_chunks(chunks):
    print("\n" + "=" * 60)
    print("CHUNK QUALITY REPORT")
    print("=" * 60)

    if not chunks:
        print("\nERROR: No chunks generated.")
        return

    # ========================================================
    # BASIC SIZE STATISTICS
    # ========================================================

    sizes = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    print(
        f"\nTotal chunks       : {len(chunks)}"
    )

    print(
        f"Average size       : "
        f"{sum(sizes) / len(sizes):.0f}"
    )

    print(
        f"Minimum size       : {min(sizes)}"
    )

    print(
        f"Maximum size       : {max(sizes)}"
    )

    # ========================================================
    # CONTENT TYPE COMBINATIONS
    # ========================================================

    combinations = {}

    for chunk in chunks:

        types = tuple(
            chunk.metadata.get(
                "content_types",
                []
            )
        )

        combinations[types] = (
            combinations.get(types, 0) + 1
        )

    print("\nContent combinations:")

    for types, count in sorted(
        combinations.items(),
        key=lambda x: str(x[0])
    ):
        print(
            f"  {types}: {count}"
        )

    # ========================================================
    # OVERSIZED CHUNKS
    # ========================================================

    oversized = [
        chunk
        for chunk in chunks
        if len(chunk.page_content)
        > MAX_CHUNK_CHARS
    ]

    print(
        f"\nOversized chunks  : "
        f"{len(oversized)}"
    )

    # --------------------------------------------------------
    # Separate atomic oversized chunks
    # --------------------------------------------------------

    atomic_oversized = [
        chunk
        for chunk in oversized
        if set(
            chunk.metadata.get(
                "content_types",
                []
            )
        ).issubset(
            {
                "table",
                "equation",
                "figure"
            }
        )
    ]

    normal_oversized = [
        chunk
        for chunk in oversized
        if chunk not in atomic_oversized
    ]

    print(
        f"  Atomic structures : "
        f"{len(atomic_oversized)}"
    )

    print(
        f"  Normal text       : "
        f"{len(normal_oversized)}"
    )

    # ========================================================
    # VERY SMALL TEXT CHUNKS
    # ========================================================

    small_text = [
        chunk
        for chunk in chunks
        if len(chunk.page_content) < 200
        and chunk.metadata.get(
            "content_types"
        ) == ["text"]
    ]

    print(
        f"\nVery small text chunks : "
        f"{len(small_text)}"
    )

    for chunk in small_text:

        print(
            f"  Chunk "
            f"{chunk.metadata.get('chunk_id')}"
            f" | "
            f"{len(chunk.page_content)} chars"
            f" | "
            f"{chunk.metadata.get('section_path')}"
        )

    # ========================================================
    # EMPTY CHUNKS
    # ========================================================

    empty = [
        chunk
        for chunk in chunks
        if not chunk.page_content.strip()
    ]

    print(
        f"\nEmpty chunks      : "
        f"{len(empty)}"
    )

    # ========================================================
    # METADATA VALIDATION
    # ========================================================

    required_metadata = [
        "chunk_id",
        "section_path",
        "content_types",
        "pages",
        "element_ids",
        "chunk_size",
    ]

    metadata_errors = []

    for chunk in chunks:

        missing = [
            key
            for key in required_metadata
            if key not in chunk.metadata
        ]

        if missing:

            metadata_errors.append(
                (
                    chunk.metadata.get(
                        "chunk_id"
                    ),
                    missing
                )
            )

    print(
        f"\nMetadata errors   : "
        f"{len(metadata_errors)}"
    )

    for chunk_id, missing in metadata_errors:

        print(
            f"  Chunk {chunk_id}: "
            f"missing {missing}"
        )

    # ========================================================
    # DUPLICATE CONTENT
    # ========================================================

    seen = {}
    duplicates = []

    for chunk in chunks:

        content = (
            chunk.page_content
            .strip()
        )

        if content in seen:

            duplicates.append(
                (
                    seen[content],
                    chunk.metadata.get(
                        "chunk_id"
                    )
                )
            )

        else:

            seen[content] = (
                chunk.metadata.get(
                    "chunk_id"
                )
            )

    print(
        f"\nDuplicate chunks : "
        f"{len(duplicates)}"
    )

    # ========================================================
    # IMAGE PATH VALIDATION
    # ========================================================

    image_errors = []

    for chunk in chunks:

        image_paths = chunk.metadata.get(
            "image_paths",
            []
        )

        for image_path in image_paths:

            if not image_path:

                image_errors.append(
                    chunk.metadata.get(
                        "chunk_id"
                    )
                )

    print(
        f"\nImage path errors : "
        f"{len(image_errors)}"
    )

    # ========================================================
    # SECTION VALIDATION
    # ========================================================

    missing_sections = [
        chunk
        for chunk in chunks
        if not chunk.metadata.get(
            "section_path"
        )
    ]

    print(
        f"\nMissing sections : "
        f"{len(missing_sections)}"
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    critical_errors = (
        len(empty)
        + len(metadata_errors)
        + len(duplicates)
    )

    print("\n" + "=" * 60)

    if critical_errors == 0:

        print(
            "VALIDATION STATUS : PASS"
        )

    else:

        print(
            "VALIDATION STATUS : REVIEW"
        )

    print("=" * 60)
    

# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "IMPROVED STRUCTURE-AWARE CHUNKING"
    )

    print("=" * 60)

    elements = load_elements()

    print(
        f"\nLoaded {len(elements)} "
        f"cleaned elements"
    )

    chunks = create_chunks(
        elements
    )

    print(
        f"\nGenerated {len(chunks)} "
        f"chunks"
    )

    # Quality validation
    validate_chunks(
        chunks
    )

    # Save chunks
    save_chunks(
        chunks
    )

    print_chunks(
        chunks
    )

if __name__ == "__main__":

    main()