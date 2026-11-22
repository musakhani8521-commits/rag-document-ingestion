from src.pipeline import RAGPipeline


def test_pipeline_ingests_markdown_document(tmp_path):
    # Create a temporary Markdown document.
    document = tmp_path / "sample.md"
    document.write_text(
        "Python is a programming language.\n\n"
        "Artificial intelligence enables intelligent tasks.",
        encoding="utf-8"
    )

    # Create the pipeline with a temporary database.
    pipeline = RAGPipeline(
        database_path=str(tmp_path / "chroma_db"),
        collection_name="test_pipeline"
    )

    # Process the document.
    chunks_count = pipeline.ingest_document(
        document,
        strategy="paragraph"
    )

    assert chunks_count > 0
    assert pipeline.vector_store.count() == chunks_count


def test_pipeline_search_returns_results(tmp_path):
    document = tmp_path / "sample.md"
    document.write_text(
        "Python is a programming language.\n\n"
        "Artificial intelligence enables intelligent tasks.",
        encoding="utf-8"
    )

    pipeline = RAGPipeline(
        database_path=str(tmp_path / "chroma_db"),
        collection_name="test_search"
    )

    pipeline.ingest_document(document)

    results = pipeline.search(
        "What is Python?",
        top_k=5
    )

    assert len(results) > 0
    assert any("Python" in item["text"] for item in results)


def test_pipeline_rejects_invalid_strategy(tmp_path):
    document = tmp_path / "sample.md"
    document.write_text(
        "This is a test document.",
        encoding="utf-8"
    )

    pipeline = RAGPipeline(
        database_path=str(tmp_path / "chroma_db"),
        collection_name="test_invalid_strategy"
    )

    try:
        pipeline.ingest_document(
            document,
            strategy="invalid"
        )
        assert False, "Expected ValueError was not raised."

    except ValueError as error:
        assert "Strategy must be" in str(error)


def test_pipeline_rejects_missing_directory(tmp_path):
    pipeline = RAGPipeline(
        database_path=str(tmp_path / "chroma_db"),
        collection_name="test_missing_directory"
    )

    try:
        pipeline.ingest_directory(
            tmp_path / "missing_directory"
        )
        assert False, "Expected FileNotFoundError was not raised."

    except FileNotFoundError:
        pass