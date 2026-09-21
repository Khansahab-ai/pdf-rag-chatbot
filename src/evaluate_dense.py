from retriever import load_vectorstore, retrieve
from evaluation_dataset import GOLD_DATASET
from evaluate import recall_at_k, reciprocal_rank


# ============================================================
# Configuration
# ============================================================

TOP_K = 5


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("DENSE RETRIEVER EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load Chroma vector store
    # --------------------------------------------------------

    vectorstore = load_vectorstore()

    # --------------------------------------------------------
    # Store metric values for all queries
    # --------------------------------------------------------

    recall_1_scores = []
    recall_3_scores = []
    recall_5_scores = []
    reciprocal_ranks = []

    # --------------------------------------------------------
    # Evaluate every query
    # --------------------------------------------------------

    for item in GOLD_DATASET:

        query = item["query"]
        relevant_chunks = item["relevant_chunks"]

        print("\n" + "-" * 70)
        print(f"Query: {query}")
        print(f"Ground truth: {relevant_chunks}")

        # ----------------------------------------------------
        # Dense retrieval
        # ----------------------------------------------------

        results = retrieve(
            vectorstore,
            query,
            k=TOP_K,
            search_type="similarity",
        )

        # ----------------------------------------------------
        # Show retrieved chunks
        # ----------------------------------------------------

        print("\nRetrieved chunks:")

        for rank, result in enumerate(results, start=1):

            document = result[0]
            score = result[1]

            chunk_id = int(
                document.metadata["chunk_id"]
            )

            section = document.metadata.get(
                "section_path"
            )

            is_relevant = (
                chunk_id in relevant_chunks
            )

            print(
                f"Rank {rank}: "
                f"Chunk {chunk_id} | "
                f"Score {score:.6f} | "
                f"Relevant: {is_relevant}"
            )

            print(f"Section: {section}")

        # ----------------------------------------------------
        # Calculate metrics for this query
        # ----------------------------------------------------

        r1 = recall_at_k(
            results,
            relevant_chunks,
            1,
        )

        r3 = recall_at_k(
            results,
            relevant_chunks,
            3,
        )

        r5 = recall_at_k(
            results,
            relevant_chunks,
            5,
        )

        rr = reciprocal_rank(
            results,
            relevant_chunks,
        )

        # ----------------------------------------------------
        # Store metrics
        # ----------------------------------------------------

        recall_1_scores.append(int(r1))
        recall_3_scores.append(int(r3))
        recall_5_scores.append(int(r5))
        reciprocal_ranks.append(rr)

        # ----------------------------------------------------
        # Show metrics for this query
        # ----------------------------------------------------

        print("\nMetrics:")

        print(f"Recall@1: {int(r1)}")
        print(f"Recall@3: {int(r3)}")
        print(f"Recall@5: {int(r5)}")
        print(f"RR      : {rr:.3f}")

    # ========================================================
    # Final Average Metrics
    # ========================================================

    recall_1 = (
        sum(recall_1_scores)
        / len(recall_1_scores)
    )

    recall_3 = (
        sum(recall_3_scores)
        / len(recall_3_scores)
    )

    recall_5 = (
        sum(recall_5_scores)
        / len(recall_5_scores)
    )

    mrr = (
        sum(reciprocal_ranks)
        / len(reciprocal_ranks)
    )

    # ========================================================
    # Final Report
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("DENSE RETRIEVER FINAL RESULTS")
    print("=" * 70)

    print(f"Recall@1 : {recall_1:.3f}")
    print(f"Recall@3 : {recall_3:.3f}")
    print(f"Recall@5 : {recall_5:.3f}")
    print(f"MRR      : {mrr:.3f}")


if __name__ == "__main__":
    main()