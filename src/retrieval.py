
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:
    """Connect query embeddings with ChromaDB similarity search."""

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore()

    def add_chunks(self, chunks, source="unknown"):
        """Generate embeddings and store document chunks."""

        if not chunks:
            return

        embeddings = self.embedding_model.generate_embeddings(chunks)

        ids = [f"{source}_{i}" for i in range(len(chunks))]

        metadatas = [
            {"source": source, "chunk_index": i}
            for i in range(len(chunks))
        ]

        self.vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

    def retrieve(self, question, top_k=5):
        """Find the most relevant chunks for a question."""

        query_embedding = (
            self.embedding_model.generate_query_embedding(question)
        )

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
