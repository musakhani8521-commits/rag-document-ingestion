
# RAG Document Ingestion Pipeline

A Python project that extracts, cleans, and prepares text from PDF, DOCX, Markdown, and HTML documents.

## Requirements
- Python 3.10 or newer
- Git

## Setup

Create a virtual environment:

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the program

```bash
python -m src.ingest data/sample_documents/example.md --owner "Training Team" --sensitivity Public
```

Save cleaned text:

```bash
python -m src.ingest data/sample_documents/example.md --output cleaned.txt
```

## Run tests

```bash
python -m pytest -v
```

Latest verified result: **12 tests passed.**

## Supported formats
- PDF — PyMuPDF
- DOCX — python-docx
- Markdown — Python file reading
- HTML — BeautifulSoup

## Features
- Document text extraction
- Text cleaning and whitespace normalization
- Document metadata generation
- SHA-256 content hash
- Error handling and logging
- Automated unit tests

## Limitations
- Scanned PDFs requiring OCR are not supported.
- Repeated-line removal uses a basic heuristic.
- Embeddings and vector database indexing are outside this project's scope.

## Project structure

```text
rag_document_ingestion/
|-- data/
|   |-- inventory_template.csv
|   |-- sample_documents/
|-- src/
|   |-- __init__.py
|   |-- ingest.py
|-- tests/
|   |-- test_ingest.py
|-- .gitignore
|-- README.md
|-- RAG_FUNDAMENTALS.md
|-- PROGRESS_REPORT_DRAFT.md
|-- requirements.txt
```
