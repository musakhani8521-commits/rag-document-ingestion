
from pathlib import Path

import pymupdf
import pytest
from docx import Document

from src.ingest import build_metadata, clean_text, extract_text


# -------------------------
# Text cleaning tests
# -------------------------

def test_clean_text_removes_repeated_lines_and_page_numbers():
    sample = """Company Confidential
First paragraph
Company Confidential
Page 1
Second paragraph
Company Confidential"""

    result = clean_text(sample)

    assert "First paragraph" in result
    assert "Second paragraph" in result
    assert "Company Confidential" not in result
    assert "Page 1" not in result


def test_clean_text_normalizes_spaces():
    sample = "  Hello    world  \n\n Next   line "

    assert clean_text(sample) == "Hello world\nNext line"


def test_clean_text_preserves_repeated_long_content():
    paragraph = (
        "This is an important paragraph that should remain "
        "in the document even when it appears multiple times."
    )
    sample = f"{paragraph}\n{paragraph}\n{paragraph}"

    result = clean_text(sample)

    assert result.count(paragraph) == 3


# -------------------------
# Document extraction tests
# -------------------------

def test_extract_markdown(tmp_path: Path):
    file_path = tmp_path / "sample.md"
    file_path.write_text("# Heading\nSome text", encoding="utf-8")

    result = extract_text(file_path)

    assert "Some text" in result


def test_extract_html(tmp_path: Path):
    file_path = tmp_path / "sample.html"
    file_path.write_text(
        "<html><body><h1>Heading</h1>"
        "<p>HTML sample text</p></body></html>",
        encoding="utf-8",
    )

    result = extract_text(file_path)

    assert "Heading" in result
    assert "HTML sample text" in result


def test_extract_docx(tmp_path: Path):
    file_path = tmp_path / "sample.docx"

    document = Document()
    document.add_paragraph("DOCX sample text")
    document.save(file_path)

    result = extract_text(file_path)

    assert "DOCX sample text" in result


def test_extract_pdf(tmp_path: Path):
    file_path = tmp_path / "sample.pdf"

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "PDF sample text")
    document.save(file_path)
    document.close()

    result = extract_text(file_path)

    assert "PDF sample text" in result


def test_empty_pdf(tmp_path: Path):
    file_path = tmp_path / "empty.pdf"

    document = pymupdf.open()
    document.new_page()
    document.save(file_path)
    document.close()

    assert extract_text(file_path) == ""


# -------------------------
# Error handling tests
# -------------------------

def test_missing_file():
    with pytest.raises(FileNotFoundError):
        extract_text("this_file_does_not_exist.pdf")


def test_unsupported_file(tmp_path: Path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError):
        extract_text(file_path)


def test_corrupted_pdf_error(tmp_path: Path):
    file_path = tmp_path / "corrupted.pdf"
    file_path.write_text(
        "This is not a valid PDF file.",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Failed to extract"):
        extract_text(file_path)


# -------------------------
# Metadata tests
# -------------------------

def test_metadata_fields():
    metadata = build_metadata(
        "guide.pdf",
        "hello",
        "IT Team",
        "Internal",
    )

    assert metadata["file_name"] == "guide.pdf"
    assert metadata["owner"] == "IT Team"
    assert metadata["type"] == "pdf"
    assert metadata["sensitivity"] == "Internal"
    assert len(metadata["sha256"]) == 64
