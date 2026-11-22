
import chromadb


class VectorStore:
    """Store and retrieve document chunks using ChromaDB."""

    def __init__(
        self,
        database_path="data/chroma_db",
        collection_name="rag_documents"
    ):
        self.client = chromadb.PersistentClient(path=database_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, chunks, embeddings, ids, metadatas=None):
        """Store document chunks, embeddings, IDs and optional metadata."""

        if not (len(chunks) == len(embeddings) == len(ids)):
            raise ValueError("Chunks, embeddings and IDs must have equal lengths.")

        if not chunks:
            return

        if metadatas is None:
            metadatas = [{"source": "unknown"} for _ in chunks]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(self, query_embedding, top_k=5):
        """Retrieve the most relevant document chunks."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []

        for i in range(len(results["ids"][0])):
            retrieved.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i]
            })

        return retrieved

    def count(self):
        """Return the number of stored document chunks."""
        return self.collection.count()
