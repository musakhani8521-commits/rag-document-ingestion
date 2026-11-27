# Task 4: Gemini-powered RAG assistant

This add-on connects the existing RAG retrieval pipeline to the **Google Gemini Developer API (Google AI Studio)**. It is an alternative to Vertex AI and must be described accurately as Gemini Developer API integration.

## Setup (Windows PowerShell)

Activate the project's virtual environment, then install the dependency:

```powershell
python -m pip install "google-genai>=3.6.1"
```

Set your API key in the current PowerShell session (replace the placeholder locally; never commit or share the key):

```powershell
$env:GEMINI_API_KEY = "YOUR_REAL_KEY"
$env:RAG_DATABASE_PATH = "data/final_evaluation_db"
$env:RAG_COLLECTION_NAME = "evaluation"
```

Run the chatbot:

```powershell
python -m src.chat
```

Type `/clear` to clear conversation history or `/exit` to close the chatbot.

## Features

- Retrieves relevant chunks through the existing `RAGPipeline.search()`.
- Sends retrieved document text to Gemini as context.
- Instructs the model to cite claims using labels such as `[S1]`.
- Instructs the model to say when the documents do not contain the answer.
- Keeps a short in-memory conversation history for follow-up questions.

## Important notes

- The implementation uses the Google AI Studio Gemini Developer API, not Vertex AI.
- Conversation history is in memory and is cleared when the program exits.
- Use non-confidential sample documents when testing the API.
- The source list identifies retrieved chunks; it does not currently provide page-number citations.
