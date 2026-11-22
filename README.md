# RAG Document Ingestion and Retrieval Pipeline

A Python-based Retrieval-Augmented Generation (RAG) document ingestion and retrieval project. It extracts text from documents, cleans and splits the text into chunks, generates local embeddings, stores the embeddings in ChromaDB, and retrieves relevant document chunks using similarity search.

## Features

- Extract text from PDF, DOCX, Markdown, and HTML files.
- Clean extracted text and normalize whitespace.
- Generate document metadata, including SHA-256 content hashes.
- Split documents using fixed-size or paragraph-based chunking.
- Generate embeddings locally using Sentence Transformers.
- Store document chunks and embeddings in ChromaDB.
- Retrieve the top 5 most relevant chunks for a question.
- Evaluate retrieval using 30 questions.
- Export individual evaluation results to CSV.
- Automated tests using pytest.

## Technologies

- Python
- Sentence Transformers
- all-MiniLM-L6-v2 embedding model
- ChromaDB
- PyMuPDF
- python-docx
- BeautifulSoup
- pytest

## Requirements

- Python 3.10 or newer
- Git
- Internet connection for installing packages and downloading the embedding model on its first use

The embedding model runs locally after it has been downloaded. No paid Google Cloud service or API key is required.

## Installation

Clone the repository:

```bash
git clone https://github.com/musakhani8521-commits/rag-document-ingestion.git
cd rag-document-ingestion
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, run this command in the current PowerShell session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Project Structure

```text
rag_document_ingestion/
|
|-- data/
|   |-- sample_documents/
|   |-- evaluation_documents/
|   |-- evaluation_questions.json
|   |-- evaluation_results.csv
|
|-- src/
|   |-- __init__.py
|   |-- ingest.py
|   |-- chunking.py
|   |-- embeddings.py
|   |-- vector_store.py
|   |-- retrieval.py
|   |-- pipeline.py
|   |-- evaluate_retrieval.py
|
|-- tests/
|   |-- test_ingest.py
|   |-- test_chunking.py
|   |-- test_vector_store.py
|   |-- test_pipeline.py
|
|-- .gitignore
|-- README.md
|-- requirements.txt
```

Generated vector database folders are created under `data/` when the pipeline runs. They are excluded from Git.

## Running the Pipeline

The integrated pipeline is implemented in `src/pipeline.py`.

To ingest all supported documents from the sample documents directory using paragraph-based chunking:

```powershell
python -c "from src.pipeline import RAGPipeline; p = RAGPipeline(); p.ingest_directory('data/sample_documents', strategy='paragraph')"
```

To use fixed-size chunking instead:

```powershell
python -c "from src.pipeline import RAGPipeline; p = RAGPipeline(); p.ingest_directory('data/sample_documents', strategy='fixed')"
```

The pipeline extracts document text, creates chunks, generates embeddings, and stores the results in ChromaDB.

## Searching Documents

After ingesting documents, retrieve the five most relevant chunks with:

```powershell
python -c "from src.pipeline import RAGPipeline; p = RAGPipeline(); results = p.search('What is artificial intelligence?', top_k=5); [print(r['metadata']['file_name'], r['text']) for r in results]"
```

The search uses the local embedding model to convert the question into a vector and ChromaDB to find similar document chunks.

## Retrieval Evaluation

The project includes 12 evaluation documents and 30 questions with expected source documents.

First, build the evaluation database:

```powershell
python -c "from src.pipeline import RAGPipeline; p = RAGPipeline(database_path='data/final_evaluation_db', collection_name='evaluation'); p.ingest_directory('data/evaluation_documents', strategy='paragraph')"
```

Then run the evaluation:

```powershell
python -m src.evaluate_retrieval
```

The evaluation reports:

- Hit@1: Percentage of questions for which the expected source is ranked first.
- Hit@5: Percentage of questions for which the expected source appears among the top five results.
- Mean Reciprocal Rank (MRR): Average reciprocal rank of the expected source.

Individual question results are saved to:

```text
data/evaluation_results.csv
```

### Latest Evaluation Results

| Metric | Result |
|---|---:|
| Questions evaluated | 30 |
| Hit@1 | 100% (30/30) |
| Hit@5 | 100% (30/30) |
| MRR | 1.0000 |

These results apply to the project's current 12-document evaluation dataset. They do not guarantee the same performance on larger or different document collections.

## Running Automated Tests

Run the complete test suite:

```powershell
python -m pytest -v
```

Latest verified result:

**29 passed, 0 failed.**

## Supported Document Formats

| Format | Extraction Library |
|---|---|
| PDF | PyMuPDF |
| DOCX | python-docx |
| Markdown | Python file reading |
| HTML | BeautifulSoup |

## Limitations

- Scanned PDFs that require OCR are not supported.
- Text cleaning uses basic heuristics.
- Retrieval evaluation currently uses a small, controlled dataset.
- The system retrieves relevant document chunks; it does not yet generate complete natural-language answers using a language model.

