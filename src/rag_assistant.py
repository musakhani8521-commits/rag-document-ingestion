"""Gemini-powered answer generation for the existing RAG pipeline."""
import os
from google import genai
from google.genai import types


class RAGAssistant:
    def __init__(self, pipeline, api_key=None, model=None, client=None, top_k=5):
        self.pipeline = pipeline
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        self.top_k = top_k
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if client is None:
            if not api_key:
                raise ValueError(
                    "Gemini API key missing. Set the GEMINI_API_KEY environment variable."
                )
            client = genai.Client(api_key=api_key)
        self.client = client

    @staticmethod
    def _format_sources(results):
        sources = []
        for index, item in enumerate(results, start=1):
            metadata = item.get("metadata") or {}
            name = metadata.get("file_name") or metadata.get("source") or "Unknown source"
            sources.append({
                "label": f"S{index}",
                "name": str(name),
                "text": str(item.get("text") or ""),
                "chunk_id": str(item.get("id") or ""),
            })
        return sources

    @staticmethod
    def _make_prompt(question, sources, history):
        if sources:
            context = "\n\n".join(
                f"[{s['label']}] Source: {s['name']}\n{s['text']}"
                for s in sources
            )
        else:
            context = "(No relevant document passages were retrieved.)"

        history_text = ""
        if history:
            history_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in history[-8:]
            )

        return f"""You are a careful assistant answering questions using the supplied documents.

Rules:
1. Answer using only information supported by the DOCUMENT CONTEXT.
2. Cite each factual claim using source labels exactly like [S1] or [S2].
3. Never invent facts, quotations, page numbers, or sources.
4. If the context does not contain enough information to answer, say:
   "I could not find that information in the provided documents."
5. Conversation history may clarify follow-up questions, but it is not evidence.
6. Keep the answer clear and concise.

CONVERSATION HISTORY:
{history_text or "(No previous conversation.)"}

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    def ask(self, question, history=None):
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        retrieved = self.pipeline.search(question, top_k=self.top_k)
        sources = self._format_sources(retrieved)
        prompt = self._make_prompt(question, sources, history or [])

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1200,
            ),
        )
        answer = (getattr(response, "text", None) or "").strip()
        if not answer:
            answer = "I could not generate an answer. Please try again."

        return {
            "answer": answer,
            "sources": [
                {"label": s["label"], "name": s["name"], "chunk_id": s["chunk_id"]}
                for s in sources
            ],
            "retrieved_count": len(retrieved),
            "abstained": "I could not find that information in the provided documents."
                         in answer,
        }

