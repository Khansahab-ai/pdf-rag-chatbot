from pathlib import Path
import json

from evaluation_dataset import GOLD_DATASET


# ============================================================
# CONFIGURATION
# ============================================================

INDEX_FILE = Path(
    r"D:\Langchain\pdf_rag_q&a\mineru_output"
    r"\attention\hybrid_auto\index_documents.json"
)


# ============================================================
# LOAD INDEX DOCUMENTS
# ============================================================

def load_documents():

    with open(
        INDEX_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("GROUND TRUTH VERIFICATION")
    print("=" * 80)

    documents = load_documents()

    print(
        f"\nLoaded {len(documents)} index documents."
    )

    # --------------------------------------------------------
    # Create chunk lookup
    # --------------------------------------------------------

    chunk_lookup = {}

    for document in documents:

        chunk_id = int(
            document["metadata"]["chunk_id"]
        )

        chunk_lookup[chunk_id] = document

    # ========================================================
    # VERIFY EVERY QUERY
    # ========================================================

    for number, item in enumerate(
        GOLD_DATASET,
        start=1,
    ):

        query = item["query"]

        relevant_chunks = item[
            "relevant_chunks"
        ]

        print("\n")
        print("-" * 80)

        print(
            f"QUERY {number}/{len(GOLD_DATASET)}"
        )

        print(
            f"Question: {query}"
        )

        print(
            f"Ground truth chunks: "
            f"{relevant_chunks}"
        )

        print("-" * 80)

        # ----------------------------------------------------
        # Check every ground-truth chunk
        # ----------------------------------------------------

        for chunk_id in relevant_chunks:

            document = chunk_lookup.get(
                chunk_id
            )

            if document is None:

                print(
                    f"\nERROR: Chunk {chunk_id} "
                    f"does not exist."
                )

                continue

            metadata = document[
                "metadata"
            ]

            section = metadata.get(
                "section_path",
                "N/A",
            )

            content = document.get(
                "page_content",
                "",
            )

            print(
                f"\nGround-truth Chunk: "
                f"{chunk_id}"
            )

            print(
                f"Section: {section}"
            )

            print(
                f"Content type: "
                f"{metadata.get('content_types')}"
            )

            print(
                "\nContent:"
            )

            print(content[:2500])

    # ========================================================
    # FINISH
    # ========================================================

    print("\n")
    print("=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()