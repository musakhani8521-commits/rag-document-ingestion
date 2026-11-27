import csv
import json
import os
import time
from pathlib import Path

from src.pipeline import RAGPipeline
from src.rag_assistant import RAGAssistant

json_path = Path("data/task4_answer_evaluation.json")
csv_path = Path("data/task4_answer_evaluation.csv")

with open(json_path, encoding="utf-8") as file:
    results = json.load(file)

database_path = os.getenv("RAG_DATABASE_PATH", "data/final_evaluation_db")
collection_name = os.getenv("RAG_COLLECTION_NAME", "evaluation")

pipeline = RAGPipeline(
    database_path=database_path,
    collection_name=collection_name,
)
assistant = RAGAssistant(pipeline)

def save():
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    fields = [
        "question", "expected_source", "answer",
        "retrieved_sources", "expected_source_retrieved",
        "abstained", "error"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

failed = [row for row in results if row.get("error")]

print(f"\nRetrying {len(failed)} failed questions.")
print("Successful answers will be preserved.\n")

for index, row in enumerate(failed, start=1):
    print(f"[{index}/{len(failed)}] {row['question']}")

    for attempt in range(3):
        try:
            response = assistant.ask(row["question"])
            sources = response["sources"]
            names = [Path(s["name"]).name for s in sources]

            row["answer"] = response["answer"]
            row["retrieved_sources"] = "; ".join(names)
            row["expected_source_retrieved"] = (
                row["expected_source"] in names
            )
            row["abstained"] = response["abstained"]
            row["error"] = ""

            print("  Success")
            break

        except Exception as exc:
            message = str(exc)
            print(f"  Attempt {attempt + 1} failed: {message[:250]}")

            if ("429" in message or "RESOURCE_EXHAUSTED" in message) and attempt < 2:
                print("  Waiting 40 seconds for the API limit to reset...")
                time.sleep(40)
            else:
                row["error"] = message
                break

    save()
    time.sleep(5)

successful = [r for r in results if not r.get("error")]
errors = len(results) - len(successful)
matched = sum(r["expected_source_retrieved"] for r in successful)

print("\n========== FINAL EVALUATION ==========")
print("Total questions:", len(results))
print("Successful responses:", len(successful))
print("Expected source retrieved:", matched)
print("Remaining errors:", errors)
print("Results saved to data/task4_answer_evaluation.csv")
print("Results saved to data/task4_answer_evaluation.json")
