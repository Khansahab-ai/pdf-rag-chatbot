from retriever import load_vectorstore
from bm25_retriever import load_documents, build_bm25
from hybrid_retriever import hybrid_retrieve
from reranker import load_reranker
from evaluation_dataset import GOLD_DATASET


# ============================================================
# CONFIGURATION
# ============================================================

CANDIDATE_K = 10

DENSE_K = 10
BM25_K = 10

FINAL_K = 5

# RRF weight values to test.
#
# alpha = RRF weight
# 1-alpha = reranker weight
#
FUSION_WEIGHTS = [
    0.25,
    0.50,
    0.75,
]


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def min_max_normalize(values):
    """
    Normalize a list of scores to the range [0, 1].

    Formula:

        normalized =
            (x - min) / (max - min)

    If all scores are identical, return 1.0
    for every item.
    """

    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return [1.0] * len(values)

    return [
        (value - minimum) / (maximum - minimum)
        for value in values
    ]


# ============================================================
# RERANK + FUSION
# ============================================================

def score_candidates(
    reranker,
    query,
    candidates,
):
    """
    Add CrossEncoder scores to hybrid candidates.

    The candidates already contain RRF scores.
    """

    if not candidates:
        return []

    pairs = []

    for candidate in candidates:

        document = candidate["document"]

        pairs.append(
            [
                query,
                document.page_content,
            ]
        )

    reranker_scores = reranker.predict(pairs)

    scored_candidates = []

    for candidate, score in zip(
        candidates,
        reranker_scores,
    ):

        result = candidate.copy()

        result["reranker_score"] = float(score)

        scored_candidates.append(result)

    return scored_candidates


def fuse_results(
    scored_candidates,
    alpha,
    k=FINAL_K,
):
    """
    Combine normalized RRF and reranker scores.

    Final score:

        alpha * normalized_RRF
        +
        (1-alpha) * normalized_Reranker
    """

    if not scored_candidates:
        return []

    # --------------------------------------------------------
    # Extract raw scores
    # --------------------------------------------------------

    rrf_scores = [
        result["rrf_score"]
        for result in scored_candidates
    ]

    reranker_scores = [
        result["reranker_score"]
        for result in scored_candidates
    ]

    # --------------------------------------------------------
    # Normalize both score types.
    # --------------------------------------------------------

    normalized_rrf = min_max_normalize(
        rrf_scores
    )

    normalized_reranker = min_max_normalize(
        reranker_scores
    )

    # --------------------------------------------------------
    # Calculate weighted final score.
    # --------------------------------------------------------

    fused_results = []

    for result, rrf_norm, reranker_norm in zip(
        scored_candidates,
        normalized_rrf,
        normalized_reranker,
    ):

        fused = result.copy()

        fused["normalized_rrf"] = rrf_norm

        fused["normalized_reranker"] = reranker_norm

        fused["fusion_score"] = (
            alpha * rrf_norm
            +
            (1 - alpha) * reranker_norm
        )

        fused_results.append(fused)

    # --------------------------------------------------------
    # Sort by final fusion score.
    # --------------------------------------------------------

    fused_results.sort(
        key=lambda result: result["fusion_score"],
        reverse=True,
    )

    return fused_results[:k]


# ============================================================
# FIND RELEVANT CHUNK RANK
# ============================================================

def find_rank(
    results,
    relevant_chunks,
):
    """
    Find the rank of the first relevant chunk.

    Returns None if it is not present.
    """

    for rank, result in enumerate(
        results,
        start=1,
    ):

        chunk_id = int(
            result["document"].metadata["chunk_id"]
        )

        if chunk_id in relevant_chunks:

            return rank

    return None


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    all_results,
):
    """
    Calculate Recall@1, Recall@3, Recall@5 and MRR.
    """

    total = len(all_results)

    hits_1 = 0
    hits_3 = 0
    hits_5 = 0

    reciprocal_ranks = []

    for results, relevant_chunks in all_results:

        rank = find_rank(
            results,
            relevant_chunks,
        )

        if rank is not None:

            if rank <= 1:
                hits_1 += 1

            if rank <= 3:
                hits_3 += 1

            if rank <= 5:
                hits_5 += 1

            reciprocal_ranks.append(
                1 / rank
            )

        else:

            reciprocal_ranks.append(0.0)

    return {
        "recall@1": hits_1 / total,
        "recall@3": hits_3 / total,
        "recall@5": hits_5 / total,
        "mrr": sum(reciprocal_ranks) / total,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("WEIGHTED RRF + RERANKER FUSION EVALUATION")
    print("=" * 90)

    # --------------------------------------------------------
    # Load vector store
    # --------------------------------------------------------

    vectorstore = load_vectorstore()

    # --------------------------------------------------------
    # Load BM25
    # --------------------------------------------------------

    print("\nLoading BM25 documents...")

    documents = load_documents()

    bm25 = build_bm25(documents)

    # --------------------------------------------------------
    # Load CrossEncoder
    # --------------------------------------------------------

    reranker = load_reranker()

    # --------------------------------------------------------
    # Store results for every weight.
    # --------------------------------------------------------

    results_by_weight = {}

    for alpha in FUSION_WEIGHTS:

        print("\n")
        print("=" * 90)
        print(
            f"TESTING ALPHA = {alpha:.2f}"
        )
        print(
            f"RRF weight = {alpha:.2f}"
        )
        print(
            f"Reranker weight = {1-alpha:.2f}"
        )
        print("=" * 90)

        weight_results = []

        # ----------------------------------------------------
        # Process every evaluation query.
        # ----------------------------------------------------

        for query_number, item in enumerate(
            GOLD_DATASET,
            start=1,
        ):

            query = item["query"]

            relevant_chunks = item[
                "relevant_chunks"
            ]

            # ------------------------------------------------
            # Generate hybrid candidates.
            # ------------------------------------------------

            candidates = hybrid_retrieve(
                vectorstore=vectorstore,
                bm25=bm25,
                documents=documents,
                query=query,
                k=CANDIDATE_K,
                dense_k=DENSE_K,
                bm25_k=BM25_K,
            )

            # ------------------------------------------------
            # Calculate reranker scores.
            # ------------------------------------------------

            scored_candidates = score_candidates(
                reranker=reranker,
                query=query,
                candidates=candidates,
            )

            # ------------------------------------------------
            # Fuse RRF + reranker.
            # ------------------------------------------------

            final_results = fuse_results(
                scored_candidates=scored_candidates,
                alpha=alpha,
                k=FINAL_K,
            )

            weight_results.append(
                (
                    final_results,
                    relevant_chunks,
                )
            )

            # ------------------------------------------------
            # Find relevant chunk rank.
            # ------------------------------------------------

            rank = find_rank(
                final_results,
                relevant_chunks,
            )

            print(
                f"Query {query_number:<2} | "
                f"Target {relevant_chunks} | "
                f"Final Rank: "
                f"{rank if rank is not None else 'N/A'}"
            )

        # ----------------------------------------------------
        # Calculate metrics for this alpha.
        # ----------------------------------------------------

        metrics = calculate_metrics(
            weight_results
        )

        results_by_weight[alpha] = metrics

        print("\nMetrics:")

        print(
            f"Recall@1 : "
            f"{metrics['recall@1']:.3f}"
        )

        print(
            f"Recall@3 : "
            f"{metrics['recall@3']:.3f}"
        )

        print(
            f"Recall@5 : "
            f"{metrics['recall@5']:.3f}"
        )

        print(
            f"MRR      : "
            f"{metrics['mrr']:.3f}"
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n\n")
    print("=" * 90)
    print("WEIGHT COMPARISON")
    print("=" * 90)

    print(
        f"{'Alpha':<10}"
        f"{'RRF %':<10}"
        f"{'Reranker %':<14}"
        f"{'R@1':<10}"
        f"{'R@3':<10}"
        f"{'R@5':<10}"
        f"{'MRR':<10}"
    )

    print("-" * 90)

    for alpha, metrics in results_by_weight.items():

        print(
            f"{alpha:<10.2f}"
            f"{alpha * 100:<10.0f}"
            f"{(1-alpha) * 100:<14.0f}"
            f"{metrics['recall@1']:<10.3f}"
            f"{metrics['recall@3']:<10.3f}"
            f"{metrics['recall@5']:<10.3f}"
            f"{metrics['mrr']:<10.3f}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()