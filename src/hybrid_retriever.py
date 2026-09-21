from src.retriever import load_vectorstore, retrieve
from src.bm25_retriever import (
    load_documents,
    build_bm25,
    retrieve_bm25,
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5

# RRF constant.
#
# Commonly a value around 60 is used.
#
# Larger values make the rank differences less aggressive.
#
RRF_K = 60


# ============================================================
# RRF SCORE
# ============================================================

def rrf_score(rank):
    """
    Calculate Reciprocal Rank Fusion contribution.

    Formula:

        RRF score = 1 / (RRF_K + rank)

    Example with RRF_K = 60:

        Rank 1:
            1 / 61

        Rank 2:
            1 / 62

        Rank 5:
            1 / 65
    """

    return 1.0 / (
        RRF_K + rank
    )


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_retrieve(
    vectorstore,
    bm25,
    documents,
    query,
    k=TOP_K,
    dense_k=10,
    bm25_k=10,
):
    """
    Retrieve documents using both:

        1. BGE-M3 dense retrieval
        2. BM25 lexical retrieval

    Then combine the rankings using RRF.
    """

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    dense_results = retrieve(
        vectorstore,
        query,
        k=dense_k,
        search_type="similarity",
    )


    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    bm25_results = retrieve_bm25(
        bm25=bm25,
        documents=documents,
        query=query,
        k=bm25_k,
    )


    # --------------------------------------------------------
    # Store RRF scores
    # --------------------------------------------------------

    fused_scores = {}

    document_data = {}


    # ========================================================
    # ADD DENSE RANKINGS
    # ========================================================

    for rank, (document, dense_score) in enumerate(dense_results,start=1):

        chunk_id = document.metadata[
            "chunk_id"
        ]

        contribution = rrf_score(
            rank
        )

        fused_scores[chunk_id] = (
            fused_scores.get(
                chunk_id,
                0.0,
            )
            + contribution
        )

        document_data[chunk_id] = {
            "document": document,
            "dense_rank": rank,
            "dense_score": dense_score,
        }


    # ========================================================
    # ADD BM25 RANKINGS
    # ========================================================

    for rank, (
        document,
        bm25_score,
    ) in enumerate(
        bm25_results,
        start=1,
    ):

        chunk_id = document.metadata[
            "chunk_id"
        ]

        contribution = rrf_score(
            rank
        )

        fused_scores[chunk_id] = (
            fused_scores.get(
                chunk_id,
                0.0,
            )
            + contribution
        )


        if chunk_id not in document_data:

            document_data[chunk_id] = {
                "document": document,
                "bm25_rank": rank,
                "bm25_score": bm25_score,
            }

        else:

            document_data[
                chunk_id
            ]["bm25_rank"] = rank

            document_data[
                chunk_id
            ]["bm25_score"] = bm25_score


    # ========================================================
    # SORT BY RRF SCORE
    # ========================================================

    ranked_chunks = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )


    # ========================================================
    # BUILD FINAL RESULTS
    # ========================================================

    results = []


    for chunk_id, score in ranked_chunks[:k]:

        data = document_data[
            chunk_id
        ]

        results.append(
            {
                "document": data["document"],
                "rrf_score": score,
                "dense_rank": data.get(
                    "dense_rank"
                ),
                "dense_score": data.get(
                    "dense_score"
                ),
                "bm25_rank": data.get(
                    "bm25_rank"
                ),
                "bm25_score": data.get(
                    "bm25_score"
                ),
            }
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
    print("HYBRID RRF SEARCH")
    print("=" * 80)

    print(
        f"\nQuery: {query}"
    )


    for rank, result in enumerate(
        results,
        start=1,
    ):

        document = result[
            "document"
        ]

        metadata = document.metadata


        print(
            "\n" + "-" * 80
        )

        print(
            f"Hybrid Rank : {rank}"
        )

        print(
            f"Chunk       : "
            f"{metadata.get('chunk_id')}"
        )

        print(
            f"Section     : "
            f"{metadata.get('section_path')}"
        )

        print(
            f"RRF Score   : "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Dense Rank  : "
            f"{result.get('dense_rank')}"
        )

        print(
            f"Dense Score : "
            f"{result.get('dense_score')}"
        )

        print(
            f"BM25 Rank   : "
            f"{result.get('bm25_rank')}"
        )

        print(
            f"BM25 Score  : "
            f"{result.get('bm25_score')}"
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

    print("=" * 80)
    print("STEP 15 - HYBRID RETRIEVAL")
    print("=" * 80)


    # --------------------------------------------------------
    # Load dense vector store
    # --------------------------------------------------------

    vectorstore = load_vectorstore()


    # --------------------------------------------------------
    # Load BM25 documents
    # --------------------------------------------------------

    documents = load_documents()


    # --------------------------------------------------------
    # Build BM25
    # --------------------------------------------------------

    bm25 = build_bm25(
        documents
    )


    # --------------------------------------------------------
    # Test queries
    # --------------------------------------------------------

    queries = [

        "What is the main architecture of the Transformer?",

        "Why does the Transformer use multi-head attention?",

        "How does scaled dot-product attention work?",

        "What is positional encoding and why is it needed?",

    ]


    # --------------------------------------------------------
    # Run hybrid retrieval
    # --------------------------------------------------------

    for query in queries:

        results = hybrid_retrieve(
            vectorstore=vectorstore,
            bm25=bm25,
            documents=documents,
            query=query,
            k=TOP_K,
            dense_k=10,
            bm25_k=10,
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