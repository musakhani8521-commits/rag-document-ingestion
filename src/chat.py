"""Interactive command-line chat for the Gemini RAG assistant."""
import os
from src.pipeline import RAGPipeline
from src.rag_assistant import RAGAssistant
from src.conversation import ChatSession


def main():
    database_path = os.getenv("RAG_DATABASE_PATH", "data/chroma_db")
    collection_name = os.getenv("RAG_COLLECTION_NAME", "rag_documents")

    pipeline = RAGPipeline(
        database_path=database_path,
        collection_name=collection_name,
    )
    assistant = RAGAssistant(pipeline)
    session = ChatSession(assistant)

    print("\nGemini RAG Chat")
    print("Ask questions about your ingested documents.")
    print("Commands: /clear clears conversation, /exit closes the chat.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not question:
            continue
        if question.lower() in {"/exit", "exit", "quit"}:
            print("Goodbye.")
            break
        if question.lower() == "/clear":
            session.clear()
            print("Conversation history cleared.\n")
            continue

        try:
            result = session.ask(question)
            print(f"\nAssistant: {result['answer']}")
            if result["sources"]:
                print("\nSources:")
                for source in result["sources"]:
                    print(f"  [{source['label']}] {source['name']} (chunk {source['chunk_id']})")
            else:
                print("\nSources: No document passages retrieved.")
            print()
        except Exception as exc:
            print(f"\nError: {exc}\n")


if __name__ == "__main__":
    main()
