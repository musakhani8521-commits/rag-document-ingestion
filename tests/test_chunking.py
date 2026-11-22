
import pytest

from src.chunking import (
    fixed_size_chunking,
    paragraph_chunking,
)


def test_fixed_size_chunking():
    text = "abcdefghij"

    chunks = fixed_size_chunking(
        text,
        chunk_size=4,
        overlap=1
    )

    assert chunks == ["abcd", "defg", "ghij", "j"]


def test_fixed_size_chunking_empty_text():
    assert fixed_size_chunking("") == []
    assert fixed_size_chunking("   ") == []


def test_fixed_size_chunking_invalid_size():
    with pytest.raises(ValueError):
        fixed_size_chunking("Hello", chunk_size=0)


def test_fixed_size_chunking_invalid_overlap():
    with pytest.raises(ValueError):
        fixed_size_chunking("Hello", chunk_size=10, overlap=10)


def test_paragraph_chunking():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks = paragraph_chunking(text, max_chunk_size=40)

    assert len(chunks) == 2
    assert "First paragraph." in chunks[0]
    assert "Second paragraph." in chunks[0]


def test_paragraph_chunking_empty_text():
    assert paragraph_chunking("") == []
    assert paragraph_chunking("   ") == []


def test_paragraph_chunking_large_paragraph():
    text = "abcdefghij"

    chunks = paragraph_chunking(text, max_chunk_size=4)

    assert chunks == ["abcd", "efgh", "ij"]
