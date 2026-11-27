"""Evaluate Gemini RAG answers against the 30-question dataset."""

import csv
import json
import os
from pathlib import Path

from src.pipeline import RAGPipeline
from src.rag_assistant import RAGAssistant


def save_results(results):
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "task4_answer_evaluation.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    fields = [
        "question",
        "expected_source",
        "answer",
        "retrieved_sources",
        "expected_source_retrieved",
        "abstained",
        "error",
    ]

    with open(
        output_dir / "task4_answer_evaluation.csv",
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


def main():
    question_file = Path("data/evaluation_questions.json")

    with open(question_file, encoding="utf-8-sig") as file:
        questions = json.load(file)

    database_path = os.getenv(
        "RAG_DATABASE_PATH", "data/final_evaluation_db"
    )
    collection_name = os.getenv("RAG_COLLECTION_NAME", "evaluation")

    pipeline = RAGPipeline(
        database_path=database_path,
        collection_name=collection_name,
    )

    assistant = RAGAssistant(pipeline)
    results = []

    print(f"\nStarting Gemini evaluation: {len(questions)} questions")
    print(f"Model: {assistant.model}\n")

    for index, item in enumerate(questions, start=1):
        question = item["question"]
        expected_source = item["expected_source"]

        print(f"[{index}/{len(questions)}] {question}")

        row = {
            "question": question,
            "expected_source": expected_source,
            "answer": "",
            "retrieved_sources": "",
            "expected_source_retrieved": False,
            "abstained": False,
            "error": "",
        }

        try:
            response = assistant.ask(question)

            answer = response["answer"]
            sources = response["sources"]

            source_names = [
                Path(source["name"]).name for source in sources
            ]

            row["answer"] = answer
            row["retrieved_sources"] = "; ".join(source_names)
            row["expected_source_retrieved"] = (
                expected_source in source_names
            )
            row["abstained"] = response["abstained"]

            print("  Expected source retrieved:",
                  row["expected_source_retrieved"])

        except Exception as exc:
            row["error"] = str(exc)
            print("  ERROR:", exc)

        results.append(row)

        # Save after every question so completed results are preserved.
        save_results(results)

    successful = [r for r in results if not r["error"]]
    correct_source = sum(
        r["expected_source_retrieved"] for r in successful
    )
    abstained = sum(r["abstained"] for r in successful)
    errors = len(results) - len(successful)

    print("\n========== EVALUATION SUMMARY ==========")
    print("Total questions:", len(results))
    print("Successful responses:", len(successful))
    print("Expected source retrieved:", correct_source)
    print("Abstained responses:", abstained)
    print("Errors:", errors)

    if successful:
        print(
            "Expected-source retrieval rate:",
            f"{correct_source / len(successful) * 100:.2f}%"
        )

    print("\nResults saved:")
    print("data/task4_answer_evaluation.csv")
    print("data/task4_answer_evaluation.json")


if __name__ == "__main__":
    main()
