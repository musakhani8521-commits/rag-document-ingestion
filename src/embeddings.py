
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Generate text embeddings using a local Sentence Transformer model."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded successfully.")

    def generate_embeddings(self, texts):
        """Convert a list of text chunks into numerical vectors."""
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings.tolist()

    def generate_query_embedding(self, query):
        """Convert one search question into a numerical vector."""
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embedding.tolist()
