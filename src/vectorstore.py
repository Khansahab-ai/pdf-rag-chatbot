import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma

from embeddings import load_embedding_model


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MINERU_DIR = BASE_DIR / "mineru_output" / "attention" / "hybrid_auto"
INDEX_FILE = MINERU_DIR / "index_documents.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

COLLECTION_NAME = "attention_is_all_you_need"


# ============================================================
# LOAD INDEX DOCUMENTS
# ============================================================

def load_index_documents(file_path: Path) -> list[dict]:

    if not file_path.exists():
        raise FileNotFoundError(
            f"Index file not found:\n{file_path}"
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as f:

        documents = json.load(f)

    if not isinstance(documents, list):
        raise ValueError(
            "index_documents.json must contain a list."
        )

    return documents


# ============================================================
# CONVERT TO LANGCHAIN DOCUMENTS
# ============================================================

def create_langchain_documents(
    index_documents: list[dict],
) -> list[Document]:

    documents = []

    for item in index_documents:

        page_content = item.get(
            "page_content",
            "",
        )

        metadata = item.get(
            "metadata",
            {},
        )

        if not page_content.strip():
            continue

        # ----------------------------------------------------
        # Sanitize metadata before sending it to Chroma
        # ----------------------------------------------------

        metadata = sanitize_metadata(metadata)

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )

    return documents


def sanitize_metadata(metadata: dict) -> dict:
    """
    Convert chunk metadata into a format accepted by Chroma.

    Chroma does not accept empty list metadata values.

    We therefore:
        - keep non-empty lists
        - remove empty lists
        - keep strings, numbers and booleans
    """

    sanitized = {}

    for key, value in metadata.items():

        # ----------------------------------------------------
        # Empty list -> remove metadata field
        # ----------------------------------------------------

        if isinstance(value, list):

            if len(value) == 0:
                continue

            sanitized[key] = value
            continue

        # ----------------------------------------------------
        # Normal metadata values
        # ----------------------------------------------------

        sanitized[key] = value

    return sanitized


# ============================================================
# CREATE CHROMA VECTOR STORE
# ============================================================

def create_vectorstore(documents: list[Document], embeddings) -> Chroma:

    print("\nCreating Chroma vector store...")
    print(
        f"Collection : {COLLECTION_NAME}"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )

    return vectorstore


# ============================================================
# VERIFY VECTOR STORE
# ============================================================

def verify_vectorstore(vectorstore: Chroma) -> None:

    collection = vectorstore._collection

    count = collection.count()

    print("\nVECTOR STORE VERIFICATION")
    print("=" * 60)

    print(
        f"Stored vectors : {count}"
    )

    print(
        f"Collection     : {COLLECTION_NAME}"
    )

    print(
        f"Database path  : {CHROMA_DIR}"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load index documents
    # --------------------------------------------------------

    print("Loading index documents...")

    index_documents = load_index_documents(
        INDEX_FILE
    )

    print(
        f"Loaded {len(index_documents)} "
        "index documents"
    )

    # --------------------------------------------------------
    # Convert to LangChain Documents
    # --------------------------------------------------------

    documents = create_langchain_documents(
        index_documents
    )

    print(
        f"Created {len(documents)} "
        "LangChain Documents"
    )

    # --------------------------------------------------------
    # Load BGE-M3
    # --------------------------------------------------------

    embeddings = load_embedding_model()

    # --------------------------------------------------------
    # Create Chroma
    # --------------------------------------------------------

    vectorstore = create_vectorstore(
        documents,
        embeddings,
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_vectorstore(
        vectorstore
    )


if __name__ == "__main__":
    main()