
# RAG Fundamentals - Simple Notes

## What is RAG?

Retrieval-Augmented Generation (RAG) is a method that allows an AI system to search documents and use relevant information to answer questions.

It retrieves useful information first and then gives that information to a language model to generate an answer.

## Key Terms

- **Token:** A small piece of text processed by a language model. It can be a word, part of a word, or punctuation.
- **Embedding:** A list of numbers representing the meaning of text. Similar meanings generally have embeddings that are close together.
- **Chunk:** A smaller piece of a long document. Documents are divided into chunks to make searching easier.
- **Vector Database:** A database that stores embeddings and searches for similar meanings.
- **Retrieval:** Finding the most relevant document chunks for a user's question.
- **Generation:** The language model creates an answer using the retrieved information.

## Six Common RAG Stages

1. **Document Collection:** Gather approved documents from their sources.
2. **Parsing and Extraction:** Read files and extract their text and useful information.
3. **Cleaning and Chunking:** Clean the extracted text and divide it into smaller pieces.
4. **Embedding and Indexing:** Convert text chunks into embeddings and store them in a searchable index.
5. **Retrieval:** Find the most relevant chunks for a user's question using semantic search.
6. **Answer Generation:** Provide the retrieved information to the language model to generate a grounded answer.

Different RAG frameworks may combine or name these stages differently. This is a practical six-stage explanation.

## Where This Week's Project Fits

This week's document ingestion pipeline covers:

- Document collection
- Parsing and text extraction
- Text cleaning
- Metadata preparation

It does not yet implement embeddings, vector search, or answer generation.
