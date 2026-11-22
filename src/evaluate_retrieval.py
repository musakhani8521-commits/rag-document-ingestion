import csv
import json
from pathlib import Path

from src.pipeline import RAGPipeline


def evaluate_retrieval():
    """Evaluate retrieval performance using 30 test questions."""

    questions_path = Path("data/evaluation_questions.json")
    results_path = Path("data/evaluation_results.csv")

    # Load evaluation questions.
    with questions_path.open("r", encoding="utf-8-sig") as file:
        questions = json.load(file)

    if not questions:
        raise ValueError("No evaluation questions found.")

    # Initialize the existing evaluation database.
    pipeline = RAGPipeline(
        database_path="data/final_evaluation_db",
        collection_name="evaluation"
    )

    if pipeline.vector_store.count() == 0:
        raise ValueError("The evaluation database is empty.")

    total = len(questions)
    hit_at_1 = 0
    hit_at_5 = 0
    reciprocal_rank_sum = 0.0

    evaluation_results = []

    print("\nRAG Retrieval Evaluation")
    print("=" * 65)

    # Evaluate every question.
    for index, item in enumerate(questions, start=1):

        question = item["question"]
        expected_source = item["expected_source"]

        # Retrieve the five most relevant chunks.
        retrieved = pipeline.search(question, top_k=5)

        # Find the rank of the expected document.
        expected_rank = None

        for rank, result in enumerate(retrieved, start=1):
            actual_source = result["metadata"].get("file_name")

            if actual_source == expected_source:
                expected_rank = rank
                break

        # Calculate evaluation metrics.
        if expected_rank == 1:
            hit_at_1 += 1

        if expected_rank is not None:
            hit_at_5 += 1
            reciprocal_rank_sum += 1 / expected_rank

        status = "PASS" if expected_rank is not None else "MISS"

        retrieved_sources = [
            result["metadata"].get("file_name", "unknown")
            for result in retrieved
        ]

        evaluation_results.append({
            "question": question,
            "expected_source": expected_source,
            "expected_rank": expected_rank or "",
            "hit_at_1": expected_rank == 1,
            "hit_at_5": expected_rank is not None,
            "retrieved_sources": "; ".join(retrieved_sources)
        })

        print(
            f"{index:02d}. {status} | "
            f"Expected: {expected_source} | "
            f"Rank: {expected_rank or 'Not found'}"
        )

    # Calculate final scores.
    hit_at_1_percentage = (hit_at_1 / total) * 100
    hit_at_5_percentage = (hit_at_5 / total) * 100
    mrr = reciprocal_rank_sum / total

    # Save individual evaluation results.
    with results_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        fieldnames = [
            "question",
            "expected_source",
            "expected_rank",
            "hit_at_1",
            "hit_at_5",
            "retrieved_sources"
        ]

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluation_results)

    # Display final evaluation summary.
    print("\n" + "=" * 65)
    print("FINAL EVALUATION RESULTS")
    print("=" * 65)

    print(f"Total questions: {total}")
    print(f"Hit@1: {hit_at_1}/{total} ({hit_at_1_percentage:.2f}%)")
    print(f"Hit@5: {hit_at_5}/{total} ({hit_at_5_percentage:.2f}%)")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    print(f"\nDetailed results saved to: {results_path}")


if __name__ == "__main__":
    evaluate_retrieval()
