from evaluation_dataset import GOLD_DATASET


# ============================================================
# Get Chunk ID
# ============================================================

def get_chunk_id(result):
    """
    Extract chunk_id from a retriever result.

    Dense / BM25 result:
        (Document, score)

    Hybrid result:
        {
            "document": Document,
            ...
        }
    """

    # Dense / BM25
    if isinstance(result, tuple):

        document = result[0]

    # Hybrid
    elif isinstance(result, dict):

        document = result["document"]

    else:

        raise TypeError(
            f"Unsupported result type: {type(result)}"
        )

    return int(document.metadata["chunk_id"])


# ============================================================
# Recall@K
# ============================================================

def recall_at_k(results, relevant_chunks, k):
    """
    Check whether a relevant chunk appears
    within the top-k results.
    """

    top_k_results = results[:k]

    retrieved_chunks = []

    for result in top_k_results:

        chunk_id = get_chunk_id(result)

        retrieved_chunks.append(chunk_id)

    return any(
        chunk_id in relevant_chunks
        for chunk_id in retrieved_chunks
    )


# ============================================================
# Reciprocal Rank
# ============================================================

def reciprocal_rank(results, relevant_chunks):
    """
    Calculate Reciprocal Rank.

    First relevant result at:

    Rank 1 -> 1.0
    Rank 2 -> 0.5
    Rank 3 -> 0.333
    """

    for rank, result in enumerate(results, start=1):

        chunk_id = get_chunk_id(result)

        if chunk_id in relevant_chunks:

            return 1 / rank

    return 0.0