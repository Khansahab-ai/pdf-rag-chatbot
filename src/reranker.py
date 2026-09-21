from sentence_transformers import CrossEncoder



# ============================================================
# CONFIGURATION
# ============================================================

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Number of hybrid candidates sent to the reranker.
CANDIDATE_K = 10

# Final number of chunks returned after reranking.
FINAL_K = 5


# ============================================================
# LOAD RERANKER
# ============================================================

def load_reranker():
    print("\nLoading reranker...")
    print(f"Model: {RERANKER_MODEL}")

    reranker = CrossEncoder(
        RERANKER_MODEL,
        device="cpu",
    )

    print("Reranker loaded successfully.")

    return reranker


# ============================================================
# RERANK
# ============================================================

def rerank(
    reranker,
    query,
    candidates,
    k=FINAL_K,
):
    """
    Rerank hybrid retrieval candidates using a CrossEncoder.

    Parameters
    ----------
    reranker:
        Loaded CrossEncoder model.

    query:
        User query.

    candidates:
        Results returned by hybrid_retrieve().

    k:
        Number of final results to return.

    Returns
    -------
    List of dictionaries ordered by reranker relevance.
    """

    if not candidates:
        return []

    # --------------------------------------------------------
    # Build query-document pairs.
    #
    # Example:
    #
    # [
    #     ["What is multi-head attention?", "chunk text 1"],
    #     ["What is multi-head attention?", "chunk text 2"],
    #     ...
    # ]
    # --------------------------------------------------------

    pairs = []

    for candidate in candidates:

        document = candidate["document"]

        pairs.append(
            [
                query,
                document.page_content,
            ]
        )

    # --------------------------------------------------------
    # CrossEncoder scores every query-document pair.
    # --------------------------------------------------------

    scores = reranker.predict(pairs)

    # --------------------------------------------------------
    # Preserve existing hybrid information while adding the
    # new reranker score.
    # --------------------------------------------------------

    reranked_results = []

    for original_rank, (candidate, score) in enumerate(
        zip(candidates, scores),
        start=1,
    ):

        result = candidate.copy()

        result["reranker_score"] = float(score)

        result["pre_rerank_rank"] = original_rank

        reranked_results.append(result)

    # --------------------------------------------------------
    # Higher CrossEncoder score = more relevant.
    # --------------------------------------------------------

    reranked_results.sort(
        key=lambda result: result["reranker_score"],
        reverse=True,
    )

    return reranked_results[:k]


# ============================================================
# PRINT RESULTS
# ============================================================

def print_reranked_results(query, results):

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    for rank, result in enumerate(results, start=1):

        document = result["document"]
        metadata = document.metadata

        print("\n" + "-" * 80)

        print(f"Final Rank       : {rank}")

        print(
            f"Before Reranking : "
            f"{result['pre_rerank_rank']}"
        )

        print(
            f"Reranker Score   : "
            f"{result['reranker_score']:.4f}"
        )

        print(
            f"RRF Score        : "
            f"{result['rrf_score']:.6f}"
        )

        print(
            f"Dense Rank       : "
            f"{result.get('dense_rank')}"
        )

        print(
            f"BM25 Rank        : "
            f"{result.get('bm25_rank')}"
        )

        print(
            f"Chunk ID         : "
            f"{metadata.get('chunk_id')}"
        )

        print(
            f"Page             : "
            f"{metadata.get('pages')}"
        )

        print(
            f"Section          : "
            f"{metadata.get('section_path')}"
        )

        print(
            f"Content Types    : "
            f"{metadata.get('content_types')}"
        )

        print("\nCONTENT:")

        print(
            document.page_content[:1200]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("CROSS-ENCODER RERANKER TEST")
    print("=" * 80)

    # --------------------------------------------------------
    # Load dense vector store.
    # --------------------------------------------------------

    vectorstore = load_vectorstore()

    # --------------------------------------------------------
    # Load BM25 documents and index.
    # --------------------------------------------------------

    print("\nLoading BM25 documents...")

    documents = load_documents()

    bm25 = build_bm25(documents)

    # --------------------------------------------------------
    # Load CrossEncoder.
    # --------------------------------------------------------

    reranker = load_reranker()

    # --------------------------------------------------------
    # Start with the query where RRF had a clear ranking issue.
    #
    # Relevant chunk = 12
    #
    # Previously:
    # Dense rank = 9
    # BM25 rank  = 5
    #
    # RRF failed to place it inside Top 5.
    # --------------------------------------------------------

    query = (
        "How does the Transformer know the position "
        "of each token in the sequence?"
    )

    # --------------------------------------------------------
    # Generate 10 hybrid candidates.
    #
    # IMPORTANT:
    # We use k=10 here instead of k=5 because the reranker
    # needs enough candidates to recover documents that RRF
    # ranked lower.
    # --------------------------------------------------------

    candidates = hybrid_retrieve(
        vectorstore=vectorstore,
        bm25=bm25,
        documents=documents,
        query=query,
        k=CANDIDATE_K,
        dense_k=10,
        bm25_k=10,
    )

    print("\nHybrid candidates before reranking:")

    for rank, result in enumerate(candidates, start=1):

        document = result["document"]

        print(
            f"Rank {rank:<2} | "
            f"Chunk {document.metadata.get('chunk_id'):<3} | "
            f"RRF {result['rrf_score']:.6f}"
        )

    # --------------------------------------------------------
    # Rerank the 10 candidates and keep final Top 5.
    # --------------------------------------------------------

    reranked_results = rerank(
        reranker=reranker,
        query=query,
        candidates=candidates,
        k=FINAL_K,
    )

    print_reranked_results(
        query,
        reranked_results,
    )


if __name__ == "__main__":
    main()