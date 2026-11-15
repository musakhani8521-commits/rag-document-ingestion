
"""Extract, clean, and describe supported documents."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import pymupdf
from bs4 import BeautifulSoup
from docx import Document

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".html", ".htm"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def extract_text(path: str | Path) -> str:
    """Extract text from a supported document."""
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {suffix}")

    try:
        if suffix == ".pdf":
            with pymupdf.open(file_path) as pdf:
                return "\n".join(page.get_text("text") for page in pdf)

        if suffix == ".docx":
            doc = Document(file_path)
            parts = [p.text for p in doc.paragraphs if p.text.strip()]

            for table in doc.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))

            return "\n".join(parts)

        if suffix in {".md", ".markdown"}:
            return file_path.read_text(encoding="utf-8")

        if suffix in {".html", ".htm"}:
            html = file_path.read_text(encoding="utf-8")
            soup = BeautifulSoup(html, "html.parser")

            for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
                tag.decompose()

            return soup.get_text(separator="\n")

    except Exception as exc:
        logger.exception("Could not extract text from %s", file_path)
        raise RuntimeError(f"Failed to extract {file_path.name}: {exc}") from exc

    raise ValueError(f"Unsupported file type: {suffix}")



def clean_text(text: str) -> str:
    """Normalize whitespace and remove repeated headers and footers."""
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    counts: dict[str, int] = {}

    for line in lines:
        normalized = line.casefold()
        counts[normalized] = counts.get(normalized, 0) + 1

    cleaned = []

    for line in lines:
        normalized = line.casefold()

        is_page_number = bool(
            re.fullmatch(r"(page\s*)?\d+(\s*of\s*\d+)?", line, re.I)
        )

        is_repeated_header = (
            counts[normalized] >= 3
            and len(line) <= 80
        )

        if not is_page_number and not is_repeated_header:
            cleaned.append(line)

    return "\n".join(cleaned).strip()



def build_metadata(
    path: str | Path,
    text: str,
    owner: str = "Unknown",
    sensitivity: str = "Unclassified",
) -> dict[str, Any]:
    """Create metadata for an ingested document."""
    file_path = Path(path)

    return {
        "source": str(file_path),
        "file_name": file_path.name,
        "owner": owner,
        "type": file_path.suffix.lower().lstrip("."),
        "sensitivity": sensitivity,
        "character_count": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def ingest_file(
    path: str | Path,
    owner: str = "Unknown",
    sensitivity: str = "Unclassified",
) -> dict[str, Any]:
    """Extract, clean, and describe a document."""
    raw_text = extract_text(path)
    cleaned = clean_text(raw_text)
    metadata = build_metadata(path, cleaned, owner, sensitivity)

    return {"text": cleaned, "metadata": metadata}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and clean document text."
    )

    parser.add_argument("file", help="Document file path")
    parser.add_argument("--owner", default="Unknown")
    parser.add_argument("--sensitivity", default="Unclassified")
    parser.add_argument("--output", default="", help="Optional output text file")

    args = parser.parse_args()

    try:
        result = ingest_file(args.file, args.owner, args.sensitivity)

        print(json.dumps(result["metadata"], indent=2))

        if args.output:
            Path(args.output).write_text(result["text"], encoding="utf-8")
            logger.info("Cleaned text saved to %s", args.output)
        else:
            print("\n--- CLEANED TEXT ---\n")
            print(result["text"])

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
