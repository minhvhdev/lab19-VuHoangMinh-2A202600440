"""Chạy 20 câu benchmark trên Flat RAG và GraphRAG, xuất comparison.csv."""
import json
import pandas as pd
from tqdm import tqdm

from . import config, flat_rag, graph_rag


def auto_score(response: str, expected_entities: list[str]) -> int:
    resp_lower = response.lower()
    hits = sum(1 for e in expected_entities if e.lower() in resp_lower)
    return 1 if hits >= max(1, len(expected_entities) // 2) else 0


def run():
    with open(config.BENCHMARK_PATH, encoding="utf-8") as f:
        questions = json.load(f)

    rows = []
    for question in tqdm(questions, desc="Benchmark"):
        try:
            flat_result = flat_rag.answer(question["question"])
        except Exception as e:
            flat_result = {"answer": f"ERROR: {e}", "prompt_tokens": 0, "completion_tokens": 0, "elapsed_sec": 0}
        try:
            graph_result = graph_rag.answer(question["question"], hops=3)
        except Exception as e:
            graph_result = {"answer": f"ERROR: {e}", "prompt_tokens": 0, "completion_tokens": 0, "elapsed_sec": 0}

        rows.append({
            "id": question["id"],
            "type": question["type"],
            "question": question["question"],
            "expected": " | ".join(question["expected_entities"]),
            "flat_rag_answer": flat_result["answer"],
            "flat_rag_score": auto_score(flat_result["answer"], question["expected_entities"]),
            "flat_rag_tokens": flat_result["prompt_tokens"] + flat_result["completion_tokens"],
            "flat_rag_time": round(flat_result["elapsed_sec"], 2),
            "graph_rag_answer": graph_result["answer"],
            "graph_rag_score": auto_score(graph_result["answer"], question["expected_entities"]),
            "graph_rag_tokens": graph_result["prompt_tokens"] + graph_result["completion_tokens"],
            "graph_rag_time": round(graph_result["elapsed_sec"], 2),
            "graph_rag_seeds": " | ".join(graph_result.get("seeds", [])),
        })

    df = pd.DataFrame(rows)
    df.to_csv(config.COMPARISON_CSV, index=False, encoding="utf-8")
    print(f"\nSaved -> {config.COMPARISON_CSV}")
    print("\n=== Summary ===")
    print(f"Flat RAG  accuracy: {df['flat_rag_score'].mean()*100:.1f}%   "
          f"avg tokens: {df['flat_rag_tokens'].mean():.0f}   "
          f"avg time: {df['flat_rag_time'].mean():.2f}s")
    print(f"GraphRAG  accuracy: {df['graph_rag_score'].mean()*100:.1f}%   "
          f"avg tokens: {df['graph_rag_tokens'].mean():.0f}   "
          f"avg time: {df['graph_rag_time'].mean():.2f}s")
    by_type = df.groupby("type")[["flat_rag_score", "graph_rag_score"]].mean() * 100
    print("\nAccuracy by question type (%):")
    print(by_type.round(1))
    return df


if __name__ == "__main__":
    run()
