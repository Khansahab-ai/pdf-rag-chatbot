from pathlib import Path
import json
import re

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INDEX_FILE = (
    BASE_DIR
    / "data"
    / "index_documents.json"
)

TOP_K = 5


# ============================================================
# LOAD INDEX DOCUMENTS
# ============================================================

def load_documents():

    print("Loading index documents...")

    with open(
        INDEX_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    documents = []

    for item in data:

        document = Document(
            page_content=item["page_content"],
            metadata=item["metadata"],
        )

        documents.append(document)

    print(
        f"Loaded {len(documents)} documents."
    )

    return documents


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str):

    """
    Convert text into tokens for BM25.

    Example:

        "Scaled Dot-Product Attention"

    becomes approximately:

        ["scaled", "dot", "product", "attention"]
    """

    text = text.lower()

    tokens = re.findall(
        r"\b\w+\b",
        text,
    )

    return tokens


# ============================================================
# BUILD BM25 INDEX
# ============================================================

def build_bm25(documents):

    print("\nBuilding BM25 index...")

    tokenized_documents = [
        tokenize(document.page_content)
        for document in documents
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    print("BM25 index built successfully.")

    return bm25


# ============================================================
# RETRIEVE
# ============================================================

def retrieve_bm25(
    bm25,
    documents,
    query: str,
    k: int = TOP_K,
):

    query_tokens = tokenize(query)

    scores = bm25.get_scores(
        query_tokens
    )

    # Sort document indexes by score.
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:k]


    results = []

    for index in ranked_indexes:

        document = documents[index]

        score = float(
            scores[index]
        )

        results.append(
            (document, score)
        )

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_results(
    query,
    results,
):

    print("\n")
    print("=" * 80)
    print("BM25 SEARCH")
    print("=" * 80)

    print(
        f"\nQuery: {query}"
    )

    print(
        f"\nTop {len(results)} results:"
    )


    for rank, (
        document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):

        metadata = document.metadata

        print(
            "\n" + "-" * 80
        )

        print(
            f"Rank    : {rank}"
        )

        print(
            f"BM25    : {score:.4f}"
        )

        print(
            f"Chunk   : "
            f"{metadata.get('chunk_id')}"
        )

        print(
            f"Pages   : "
            f"{metadata.get('pages')}"
        )

        print(
            f"Section : "
            f"{metadata.get('section_path')}"
        )

        print(
            f"Type    : "
            f"{metadata.get('content_types')}"
        )

        print("\nContent:")

        print(
            document.page_content[:1000]
        )

    print(
        "\n" + "=" * 80
    )


# ============================================================
# MAIN
# ============================================================

def main():

    documents = load_documents()

    bm25 = build_bm25(
        documents
    )


    queries = [

        "What is the main architecture of the Transformer?",

        "Why does the Transformer use multi-head attention?",

        "How does scaled dot-product attention work?",

        "What is positional encoding and why is it needed?",

    ]


    for query in queries:

        results = retrieve_bm25(
            bm25=bm25,
            documents=documents,
            query=query,
            k=TOP_K,
        )

        print_results(
            query,
            results,
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()