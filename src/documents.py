import json
import re
from pathlib import Path

from langchain_core.documents import Document


# ============================================================
# PATHS
# ============================================================

MINERU_DIR = Path(
    r"D:\Langchain\pdf_rag_q&a\mineru_output\attention\hybrid_auto"
)

INPUT_FILE = MINERU_DIR / "cleaned_elements.json"


# ============================================================
# HEADING HELPERS
# ============================================================

def get_heading_number(text):
    """
    Extract heading numbering.

    Examples:

        3 Model Architecture
            -> 3

        3.2 Attention
            -> 3.2

        3.2.1 Scaled Dot-Product Attention
            -> 3.2.1

        Abstract
            -> None
    """

    match = re.match(
        r"^\s*(\d+(?:\.\d+)*)\s+",
        text
    )

    if match:
        return match.group(1)

    return None


def get_heading_level(text):
    """
    Determine hierarchy level.

    3       -> 1
    3.2     -> 2
    3.2.1   -> 3

    Non-numbered headings such as Abstract,
    References -> level 1
    """

    number = get_heading_number(text)

    if number is None:
        return 1

    return number.count(".") + 1


# ============================================================
# LOAD
# ============================================================

def load_elements():

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# CREATE DOCUMENTS
# ============================================================

def create_documents(elements):

    documents = []

    current_section_path = []

    document_title = None

    # Stable ID for retrieval/debugging
    document_element_id = 0

    for original_index, element in enumerate(elements):

        content_type = element.get(
            "content_type"
        )

        text = element.get(
            "text",
            ""
        ).strip()

        # ----------------------------------------------------
        # DOCUMENT TITLE
        # ----------------------------------------------------

        if content_type == "document_title":

            document_title = text

            continue

        # ----------------------------------------------------
        # EMPTY ELEMENT
        # ----------------------------------------------------

        if not text and content_type != "table":

            continue

        # ----------------------------------------------------
        # HEADING
        # ----------------------------------------------------

        if content_type == "heading":

            heading = text

            if not heading:
                continue

            level = get_heading_level(
                heading
            )

            # Keep only parent headings
            current_section_path = (
                current_section_path[:level - 1]
            )

            current_section_path.append(
                heading
            )

            continue

        # ----------------------------------------------------
        # COMMON METADATA
        # ----------------------------------------------------

        metadata = {
            "source": "attention.pdf",

            "page": element.get(
                "page"
            ),

            "content_type": content_type,

            "element_id": document_element_id,

            "source_element_index": original_index,

            "section_path": (
                current_section_path.copy()
            ),
        }

        # Add title to every document
        if document_title:

            metadata["document_title"] = (
                document_title
            )

        # ----------------------------------------------------
        # FIGURE
        # ----------------------------------------------------

        if content_type == "figure":

            caption = element.get(
                "caption",
                ""
            ).strip()

            page_content = (
                caption
                if caption
                else text
            )

            image_path = element.get(
                "image_path"
            )

            if image_path:

                metadata["image_path"] = str(
                    MINERU_DIR / image_path
                )

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata
                )
            )

        # ----------------------------------------------------
        # EQUATION
        # ----------------------------------------------------

        elif content_type == "equation":

            metadata["element_type"] = (
                "display_equation"
            )

            image_path = element.get(
                "image_path"
            )

            if image_path:

                metadata["image_path"] = str(
                    MINERU_DIR / image_path
                )

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )

        # ----------------------------------------------------
        # TABLE
        # ----------------------------------------------------

        elif content_type == "table":

            caption = element.get(
                "caption",
                ""
            ).strip()

            table_html = element.get(
                "html",
                ""
            ).strip()

            # Preserve caption and complete table
            page_content = (
                f"{caption}\n\n"
                f"{table_html}"
            ).strip()

            metadata["table_type"] = (
                element.get(
                    "table_type"
                )
            )

            image_path = element.get(
                "image_path"
            )

            if image_path:

                metadata["image_path"] = str(
                    MINERU_DIR / image_path
                )

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata
                )
            )

        # ----------------------------------------------------
        # NORMAL TEXT
        # ----------------------------------------------------

        elif content_type == "text":

            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )

        # ----------------------------------------------------
        # SAFETY
        # ----------------------------------------------------

        else:

            print(
                f"Skipping unsupported "
                f"content type: {content_type}"
            )

        document_element_id += 1

    return documents, document_title


# ============================================================
# PRINT DOCUMENT
# ============================================================

def print_document(index, doc):

    print(
        f"\n--- Document {index} ---"
    )

    print(
        "Content type:",
        doc.metadata.get(
            "content_type"
        )
    )

    print(
        "Page:",
        doc.metadata.get(
            "page"
        )
    )

    print(
        "Section:",
        doc.metadata.get(
            "section_path"
        )
    )

    print(
        "Element ID:",
        doc.metadata.get(
            "element_id"
        )
    )

    print(
        "Content:",
        doc.page_content[:300]
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(documents):

    counts = {}

    for doc in documents:

        content_type = doc.metadata.get(
            "content_type"
        )

        counts[content_type] = (
            counts.get(
                content_type,
                0
            ) + 1
        )

    print(
        "\n" + "=" * 60
    )

    print(
        "DOCUMENT SUMMARY"
    )

    print(
        "=" * 60
    )

    for content_type, count in sorted(
        counts.items()
    ):

        print(
            f"{content_type:20} : {count}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "LANGCHAIN DOCUMENT CREATION"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    elements = load_elements()

    print(
        f"\nLoaded {len(elements)} "
        f"cleaned elements"
    )

    # --------------------------------------------------------
    # Create documents
    # --------------------------------------------------------

    documents, document_title = (
        create_documents(elements)
    )

    print(
        f"Document title: "
        f"{document_title}"
    )

    print(
        f"Created {len(documents)} "
        f"LangChain Documents"
    )

    # --------------------------------------------------------
    # Samples
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "SAMPLE DOCUMENTS"
    )

    print(
        "=" * 60
    )

    for i, doc in enumerate(
        documents[:10]
    ):

        print_document(
            i,
            doc
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        documents
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()