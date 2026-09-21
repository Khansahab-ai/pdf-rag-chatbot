"""
Step 12 - Retrieval Evaluation

Evaluates the current dense similarity retriever.

Metrics:
    Recall@1
    Recall@3
    Recall@5
    MRR
"""

from retriever import load_vectorstore, retrieve

# ============================================================
# GOLDEN DATASET
# ============================================================
#
# We identify relevant results using section names rather than
# hard-coded chunk numbers.
#
# This is important because chunk IDs may change later when
# we improve our chunking strategy.
#
# ============================================================


GOLD_DATASET = [

    {
        "query": "What is the main architecture of the Transformer?",
        "expected_terms": ["Model Architecture"],
    },

    {
        "query": "Why does the Transformer use multi-head attention?",
        "expected_terms": ["Multi-Head Attention"],
    },

    {
        "query": "How does scaled dot-product attention work?",
        "expected_terms": ["Scaled Dot-Product Attention"],
    },

    {
        "query": "What is positional encoding and why is it needed?",
        "expected_terms": ["Positional Encoding"],
    },
]


# ============================================================
# RELEVANCE CHECK
# ============================================================

def is_relevant(document, expected_terms):
    """
    Check whether a retrieved document is relevant.

    We look at:
        1. section_path
        2. document text

    We don't depend on chunk numbers.
    """

    metadata = document.metadata

    section_path = str(
        metadata.get("section_path", "")
    )

    text = document.page_content

    searchable_text = (
        section_path + " " + text
    ).lower()

    return all(
        term.lower() in searchable_text
        for term in expected_terms
    )


# ============================================================
# RECIPROCAL RANK
# ============================================================

def reciprocal_rank(relevant_ranks):
    """
    Calculate reciprocal rank.

    Example:

        relevant result at rank 1
        -> 1 / 1 = 1.0

        relevant result at rank 2
        -> 1 / 2 = 0.5

        relevant result at rank 5
        -> 1 / 5 = 0.2

        no relevant result
        -> 0
    """

    if not relevant_ranks:
        return 0.0

    first_relevant_rank = relevant_ranks[0]

    return 1.0 / first_relevant_rank


# ============================================================
# MAIN
# ============================================================

def main():

    vectorstore = load_vectorstore()

    print("=" * 80)
    print("STEP 12 - RETRIEVAL EVALUATION")
    print("=" * 80)

    total_queries = len(GOLD_DATASET)

    recall_1_hits = 0
    recall_3_hits = 0
    recall_5_hits = 0

    reciprocal_ranks = []


    # ========================================================
    # EVALUATE EVERY QUERY
    # ========================================================

    for query_number, item in enumerate(
        GOLD_DATASET,
        start=1
    ):

        query = item["query"]
        expected_terms = item["expected_terms"]


        print("\n")
        print("-" * 80)

        print(
            f"QUERY {query_number}/{total_queries}"
        )

        print(
            f"Question : {query}"
        )

        print(
            f"Expected : {', '.join(expected_terms)}"
        )

        print("-" * 80)


        # ----------------------------------------------------
        # Retrieve top 5 documents
        # ----------------------------------------------------

        results = retrieve(
            vectorstore=vectorstore,
            query=query,
            k=5,
            search_type="similarity",
        )


        relevant_ranks = []


        # ----------------------------------------------------
        # Inspect retrieved documents
        # ----------------------------------------------------

        for rank, (document, score) in enumerate(
            results,
            start=1
        ):
    
            section = document.metadata.get(
                "section_path",
                "N/A"
            )

            chunk_id = document.metadata.get(
                "chunk_id",
                "N/A"
            )

            relevant = is_relevant(
                document,
                expected_terms
            )

            if relevant:
                relevant_ranks.append(rank)

            status = "✓ RELEVANT" if relevant else ""

            print(
                f"Rank {rank}: "
                f"chunk={chunk_id} | "
                f"section={section} "
                f"{status}"
            )

        # ----------------------------------------------------
        # Recall@1
        # ----------------------------------------------------

        if any(
            rank <= 1
            for rank in relevant_ranks
        ):
            recall_1_hits += 1


        # ----------------------------------------------------
        # Recall@3
        # ----------------------------------------------------

        if any(
            rank <= 3
            for rank in relevant_ranks
        ):
            recall_3_hits += 1


        # ----------------------------------------------------
        # Recall@5
        # ----------------------------------------------------

        if any(
            rank <= 5
            for rank in relevant_ranks
        ):
            recall_5_hits += 1


        # ----------------------------------------------------
        # MRR
        # ----------------------------------------------------

        rr = reciprocal_rank(
            relevant_ranks
        )

        reciprocal_ranks.append(rr)


        # ----------------------------------------------------
        # Query result
        # ----------------------------------------------------

        if relevant_ranks:

            print(
                f"\nRelevant ranks : "
                f"{relevant_ranks}"
            )

            print(
                f"First relevant : "
                f"Rank {relevant_ranks[0]}"
            )

            print(
                f"Reciprocal rank: "
                f"{rr:.3f}"
            )

        else:

            print(
                "\nRelevant ranks : NONE"
            )

            print(
                "Reciprocal rank: 0.000"
            )


    # ========================================================
    # FINAL METRICS
    # ========================================================

    recall_at_1 = (
        recall_1_hits /
        total_queries
    )

    recall_at_3 = (
        recall_3_hits /
        total_queries
    )

    recall_at_5 = (
        recall_5_hits /
        total_queries
    )

    mrr = (
        sum(reciprocal_ranks) /
        total_queries
    )


    print("\n")

    print("=" * 80)
    print("FINAL RETRIEVAL METRICS")
    print("=" * 80)

    print(
        f"Total queries : {total_queries}"
    )

    print(
        f"Recall@1      : {recall_at_1:.3f}"
    )

    print(
        f"Recall@3      : {recall_at_3:.3f}"
    )

    print(
        f"Recall@5      : {recall_at_5:.3f}"
    )

    print(
        f"MRR           : {mrr:.3f}"
    )

    print("=" * 80)


    # ========================================================
    # HUMAN READABLE INTERPRETATION
    # ========================================================

    print("\nINTERPRETATION")
    print("-" * 80)

    print(
        f"Recall@1 = {recall_at_1:.1%}"
    )

    print(
        "Percentage of questions where a relevant "
        "document appeared at rank 1."
    )


    print(
        f"\nRecall@3 = {recall_at_3:.1%}"
    )

    print(
        "Percentage of questions where a relevant "
        "document appeared within the top 3."
    )


    print(
        f"\nRecall@5 = {recall_at_5:.1%}"
    )

    print(
        "Percentage of questions where a relevant "
        "document appeared within the top 5."
    )


    print(
        f"\nMRR = {mrr:.3f}"
    )

    print(
        "Measures how highly the first relevant "
        "document was ranked."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()