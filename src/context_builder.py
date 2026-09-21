from typing import List, Dict


# ============================================================
# CONFIGURATION
# ============================================================

MAX_CONTEXT_CHARS = 12000


# ============================================================
# FORMAT ONE DOCUMENT
# ============================================================

def format_document(
    document,
    source_number: int,
) -> str:
    """
    Convert one LangChain Document into a structured
    context block for the LLM.
    """

    metadata = document.metadata

    chunk_id = metadata.get(
        "chunk_id",
        "unknown",
    )

    pages = metadata.get(
        "pages",
        [],
    )

    section_path = metadata.get(
        "section_path",
        [],
    )

    content_types = metadata.get(
        "content_types",
        [],
    )

    # --------------------------------------------------------
    # Format page information
    # --------------------------------------------------------

    if isinstance(pages, list):

        page_text = ", ".join(
            str(page)
            for page in pages
        )

    else:

        page_text = str(pages)

    # --------------------------------------------------------
    # Format section hierarchy
    # --------------------------------------------------------

    if isinstance(section_path, list):

        section_text = " > ".join(
            str(section)
            for section in section_path
        )

    else:

        section_text = str(section_path)

    # --------------------------------------------------------
    # Format content types
    # --------------------------------------------------------

    if isinstance(content_types, list):

        content_type_text = ", ".join(
            str(content_type)
            for content_type in content_types
        )

    else:

        content_type_text = str(
            content_types
        )

    # --------------------------------------------------------
    # Build context block
    # --------------------------------------------------------

    context_block = (
        f"[SOURCE {source_number}]\n"
        f"Chunk ID: {chunk_id}\n"
        f"Page: {page_text}\n"
        f"Section: {section_text}\n"
        f"Content Type: {content_type_text}\n"
        f"Content:\n"
        f"{document.page_content.strip()}\n"
        f"[/SOURCE {source_number}]"
    )

    return context_block


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    results,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Convert retrieved/reranked results into one
    structured context string.
    """

    if not results:

        return ""

    context_blocks = []

    current_length = 0

    source_number = 1

    for result in results:

        document = result["document"]

        block = format_document(
            document=document,
            source_number=source_number,
        )

        block_length = len(block)

        # ----------------------------------------------------
        # Stop if adding this source would exceed the
        # context character limit.
        # ----------------------------------------------------

        if (
            current_length + block_length
            > max_chars
        ):

            break

        context_blocks.append(block)

        current_length += block_length

        source_number += 1

    # --------------------------------------------------------
    # Separate sources clearly.
    # --------------------------------------------------------

    return "\n\n".join(
        context_blocks
    )


# ============================================================
# PRINT CONTEXT
# ============================================================

def print_context(context: str):
    """
    Print the final context for inspection.
    """

    print("\n")
    print("=" * 80)
    print("FINAL LLM CONTEXT")
    print("=" * 80)

    print(context)

    print("\n" + "=" * 80)

    print(
        f"Context characters: {len(context)}"
    )

    print("=" * 80)


# ============================================================
# TEST CONTEXT BUILDER
# ============================================================

if __name__ == "__main__":

    from retriever import load_vectorstore

    from bm25_retriever import (
        load_documents,
        build_bm25,
    )

    from hybrid_retriever import hybrid_retrieve

    from reranker import load_reranker


    query = (
        "What is positional encoding "
        "and why is it needed?"
    )


    # --------------------------------------------------------
    # Load retrieval components
    # --------------------------------------------------------

    vectorstore = load_vectorstore()

    documents = load_documents()

    bm25 = build_bm25(documents)

    reranker = load_reranker()


    # --------------------------------------------------------
    # Retrieve candidates
    # --------------------------------------------------------

    candidates = hybrid_retrieve(
        vectorstore=vectorstore,
        bm25=bm25,
        documents=documents,
        query=query,
        k=10,
        dense_k=10,
        bm25_k=10,
    )


    # --------------------------------------------------------
    # Score candidates with reranker
    # --------------------------------------------------------

    pairs = [
        [
            query,
            candidate["document"].page_content,
        ]
        for candidate in candidates
    ]

    reranker_scores = reranker.predict(
        pairs
    )


    scored_candidates = []

    for candidate, score in zip(
        candidates,
        reranker_scores,
    ):

        result = candidate.copy()

        result["reranker_score"] = float(
            score
        )

        scored_candidates.append(result)


    # --------------------------------------------------------
    # Sort by reranker score.
    #
    # This test is only checking whether the context
    # builder correctly receives and formats retrieved
    # documents.
    # --------------------------------------------------------

    scored_candidates.sort(
        key=lambda result: result[
            "reranker_score"
        ],
        reverse=True,
    )


    # Keep top 5 for the context test.

    final_results = scored_candidates[:5]


    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        final_results
    )


    # --------------------------------------------------------
    # Print context
    # --------------------------------------------------------

    print_context(context)