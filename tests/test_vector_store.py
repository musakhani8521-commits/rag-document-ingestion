
import pytest

from src.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path):
    """Create a separate temporary ChromaDB for each test."""
    return VectorStore(
        database_path=str(tmp_path / "chroma_test"),
        collection_name="test_collection"
    )


def test_add_documents(vector_store):
    chunks = [
        "Python is a programming language.",
        "Artificial intelligence enables intelligent tasks."
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]

    ids = ["doc1", "doc2"]

    vector_store.add_documents(chunks, embeddings, ids)

    assert vector_store.count() == 2


def test_search_returns_relevant_document(vector_store):
    chunks = [
        "Python is a programming language.",
        "Artificial intelligence enables intelligent tasks."
    ]

    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ]

    vector_store.add_documents(
        chunks,
        embeddings,
        ["doc1", "doc2"]
    )

    results = vector_store.search(
        query_embedding=[0.0, 1.0, 0.0],
        top_k=1
    )

    assert len(results) == 1
    assert results[0]["text"] == chunks[1]


def test_search_empty_database(vector_store):
    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0]
    )

    assert results == []


def test_invalid_document_lengths(vector_store):
    with pytest.raises(ValueError):
        vector_store.add_documents(
            chunks=["Python"],
            embeddings=[[1.0, 0.0, 0.0]],
            ids=["doc1", "doc2"]
        )


def test_invalid_top_k(vector_store):
    with pytest.raises(ValueError):
        vector_store.search(
            query_embedding=[1.0, 0.0, 0.0],
            top_k=0
        )


def test_metadata_is_stored(vector_store):
    vector_store.add_documents(
        chunks=["Python is a programming language."],
        embeddings=[[1.0, 0.0, 0.0]],
        ids=["doc1"],
        metadatas=[{"source": "python.txt"}]
    )

    results = vector_store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1
    )

    assert results[0]["metadata"]["source"] == "python.txt"
