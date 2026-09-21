from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-m3"


def load_embedding_model():
    """
    Load the BGE-M3 embedding model.
    """

    print("Loading embedding model...")
    print(f"Model: {MODEL_NAME}")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        },
    )

    print("Embedding model loaded successfully.")

    return embeddings


def main():

    embeddings = load_embedding_model()

    # --------------------------------------------------------
    # Test document embedding
    # --------------------------------------------------------

    test_text = (
        "The Transformer is a neural network architecture "
        "based entirely on attention mechanisms."
    )

    vector = embeddings.embed_query(test_text)

    print("\nEMBEDDING TEST")
    print("=" * 60)

    print(
        f"Embedding dimension : {len(vector)}"
    )

    print(
        "First 10 values      : "
        f"{vector[:10]}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()