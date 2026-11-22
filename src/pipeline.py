
from pathlib import Path

from src.ingest import ingest_file, SUPPORTED_EXTENSIONS
from src.chunking import fixed_size_chunking, paragraph_chunking
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class RAGPipeline:
    """Complete document ingestion and retrieval pipeline."""

    def __init__(
        self,
        database_path="data/chroma_db",
        collection_name="rag_documents"
    ):
        """Initialize the embedding model and vector database."""

        self.embedding_model = EmbeddingModel()

        self.vector_store = VectorStore(
            database_path=database_path,
            collection_name=collection_name
        )

    def ingest_document(
        self,
        file_path,
        strategy="paragraph",
        owner="Unknown",
        sensitivity="Unclassified"
    ):
        """Extract, clean, chunk, embed and store one document."""

        file_path = Path(file_path)

        # 1. Extract and clean document text.
        document = ingest_file(
            file_path,
            owner=owner,
            sensitivity=sensitivity
        )

        text = document["text"]
        metadata = document["metadata"]

        if not text:
            print(f"No text found in {file_path.name}")
            return 0

        # 2. Split the text into chunks.
        if strategy == "fixed":
            chunks = fixed_size_chunking(text)

        elif strategy == "paragraph":
            chunks = paragraph_chunking(text)

        else:
            raise ValueError(
                "Strategy must be 'fixed' or 'paragraph'."
            )

        if not chunks:
            return 0

        # 3. Generate embeddings for all chunks.
        embeddings = self.embedding_model.generate_embeddings(chunks)

        # 4. Generate stable document chunk IDs.
        document_hash = metadata["sha256"]

        ids = [
            f"{document_hash}_{i}"
            for i in range(len(chunks))
        ]

        # 5. Prepare metadata for every chunk.
        metadatas = [
            {
                "source": metadata["source"],
                "file_name": metadata["file_name"],
                "owner": metadata["owner"],
                "type": metadata["type"],
                "sensitivity": metadata["sensitivity"],
                "sha256": document_hash,
                "chunk_index": i,
                "chunk_strategy": strategy
            }
            for i in range(len(chunks))
        ]

        # 6. Store chunks, embeddings and metadata.
        self.vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        print(
            f"Processed: {file_path.name} | "
            f"Chunks: {len(chunks)}"
        )

        return len(chunks)

    def ingest_directory(
        self,
        directory,
        strategy="paragraph"
    ):
        """Process all supported documents in a directory."""

        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory not found: {directory}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Not a directory: {directory}"
            )

        total_chunks = 0
        processed_files = 0

        for file_path in sorted(directory.rglob("*")):

            if (
                file_path.is_file()
                and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                chunks_count = self.ingest_document(
                    file_path,
                    strategy=strategy
                )

                total_chunks += chunks_count
                processed_files += 1

        print("\nIngestion summary")
        print("-----------------")
        print(f"Documents processed: {processed_files}")
        print(f"Total chunks created: {total_chunks}")
        print(f"Total chunks stored: {self.vector_store.count()}")

        return total_chunks

    def search(self, question, top_k=5):
        """Retrieve the most relevant chunks for a question."""

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        query_embedding = (
            self.embedding_model.generate_query_embedding(question)
        )

        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )
