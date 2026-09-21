from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings import load_embedding_model

# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DIR = Path(
    r"D:\Langchain\pdf_rag_q&a\chroma_db"
)

COLLECTION_NAME = "attention_is_all_you_need"

TOP_K = 5


# ============================================================
# LOAD VECTOR STORE
# ============================================================

def load_vectorstore():

    print("Loading embedding model...")

    embeddings = load_embedding_model()

    print("\nLoading Chroma vector store...")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    print("Vector store loaded successfully.")

    return vectorstore


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve(
    vectorstore,
    query: str,
    k: int = TOP_K,
    search_type: str = "similarity",
):
    """
    Retrieve relevant documents.

    search_type:
        similarity -> standard vector similarity
        mmr        -> Maximal Marginal Relevance
    """

    if search_type == "similarity":

        results = vectorstore.similarity_search_with_score(
            query,
            k=k,
        )

        return results

    elif search_type == "mmr":

        documents = vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=15,
            lambda_mult=0.5,
        )

        # MMR doesn't return scores through this API.
        return [
            (document, None)
            for document in documents
        ]

    else:

        raise ValueError(
            f"Unknown search type: {search_type}"
        )

# ============================================================
# DISPLAY RESULTS
# ============================================================

def print_results(
    query: str,
    results,
):

    print("\n")
    print("=" * 80)
    print("QUERY")
    print("=" * 80)

    print(query)

    print("\n")
    print("=" * 80)
    print(f"TOP {len(results)} RESULTS")
    print("=" * 80)

    for rank, (document, score) in enumerate(results,start=1):

        metadata = document.metadata

        print("\n" + "-" * 80)

        print(
            f"Rank       : {rank}"
        )

        if score is not None:
            print(f"Score      : {score:.4f}")
        else:
            print("Score      : N/A (MMR)")

        print(
            f"Chunk ID    : "
            f"{metadata.get('chunk_id')}"
        )

        print(
            f"Page       : "
            f"{metadata.get('pages')}"
        )

        print(
            f"Section    : "
            f"{metadata.get('section_path')}"
        )

        print(
            f"Type       : "
            f"{metadata.get('content_types')}"
        )

        print("\nContent:")

        print(
            document.page_content[:1500]
        )

    print("\n" + "=" * 80)


# ============================================================
# MAIN
# ============================================================

def main():

    vectorstore = load_vectorstore()

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
    # Run retrieval
    # --------------------------------------------------------

    for query in queries:

        print("\n\n")
        print("#" * 80)
        print("SIMILARITY SEARCH")
        print("#" * 80)

        similarity_results = retrieve(
            vectorstore,
            query,
            k=TOP_K,
            search_type="similarity",
        )

        print_results(
            query,
            similarity_results,
        )

        print("\n\n")
        print("#" * 80)
        print("MMR SEARCH")
        print("#" * 80)

        mmr_results = retrieve(
            vectorstore,
            query,
            k=TOP_K,
            search_type="mmr",
        )

        print_results(
            query,
            mmr_results,
        )

if __name__ == "__main__":
    main()