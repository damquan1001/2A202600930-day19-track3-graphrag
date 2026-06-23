"""
Step 5: Evaluation — Compare Flat RAG vs GraphRAG on benchmark questions.
"""

import time
import json
import os
from tabulate import tabulate

from config import BENCHMARK_QUESTIONS, OUTPUT_DIR
from utils import ensure_dir


def evaluate_answers(question: str, flat_answer: str, graph_answer: str) -> dict:
    """
    Evaluate answers: compare Flat RAG vs GraphRAG.
    Returns a dict with evaluation results.
    """
    # Simple heuristic evaluation
    flat_len = len(flat_answer) if flat_answer else 0
    graph_len = len(graph_answer) if graph_answer else 0

    has_flat_error = "no relevant" in (flat_answer or "").lower() or "insufficient" in (flat_answer or "").lower()
    has_graph_error = "no relevant" in (graph_answer or "").lower() or "insufficient" in (graph_answer or "").lower() or "not found" in (graph_answer or "").lower()

    # Determine if one is clearly better
    # We'll use an LLM judge for more accurate comparison
    winner = "unknown"
    if has_flat_error and not has_graph_error:
        winner = "GraphRAG"
    elif has_graph_error and not has_flat_error:
        winner = "Flat RAG"
    elif has_flat_error and has_graph_error:
        winner = "both_insufficient"

    return {
        "question": question,
        "flat_answer_length": flat_len,
        "graph_answer_length": graph_len,
        "flat_insufficient": has_flat_error,
        "graph_insufficient": has_graph_error,
        "winner": winner,
    }


def llm_judge(question: str, flat_answer: str, graph_answer: str) -> dict:
    """Use LLM as judge to compare answers."""
    from utils import llm_call, get_tracker

    judge_prompt = f"""You are evaluating two answers to the same question about the electric vehicle industry.

Question: {question}

Answer A (Flat RAG — simple document retrieval):
{flat_answer}

Answer B (GraphRAG — knowledge graph traversal):
{graph_answer}

Evaluate which answer is more accurate, comprehensive, and specific. Consider:
1. Factual accuracy (no hallucinations)
2. Relevance to the question
3. Specificity (mentions specific entities, data, relationships)
4. Completeness

Return a JSON with:
- "winner": "A" (Flat RAG) or "B" (GraphRAG) or "TIE"
- "reason": brief explanation (1-2 sentences)
- "score_a": 1-10
- "score_b": 1-10

Return ONLY the JSON object."""

    try:
        result, pt, ct, cost, duration = llm_call(prompt=judge_prompt)
        get_tracker().add_usage("Evaluation: LLM judge", pt, ct, cost, duration)

        # Parse JSON
        import re, json
        json_match = re.search(r"\{.*\}", result, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            return parsed
        return {"winner": "unknown", "reason": "Failed to parse judge response", "score_a": 0, "score_b": 0}
    except Exception as e:
        return {"winner": "unknown", "reason": str(e), "score_a": 0, "score_b": 0}


def run_evaluation(flat_rag_fn, graph_rag_fn, G, questions: list[str] = None) -> list[dict]:
    """
    Run full evaluation comparing Flat RAG vs GraphRAG.

    Args:
        flat_rag_fn: function(question) -> dict
        graph_rag_fn: function(G, question) -> dict
        G: NetworkX graph
        questions: list of questions (defaults to config.BENCHMARK_QUESTIONS)
    """
    if questions is None:
        questions = BENCHMARK_QUESTIONS

    results = []

    print("\n" + "=" * 70)
    print("EVALUATION: Flat RAG vs GraphRAG")
    print("=" * 70)

    for i, question in enumerate(questions):
        print(f"\n--- Question {i+1}: {question[:80]}...")

        # Flat RAG
        print("\n[Flat RAG]")
        start = time.time()
        flat_result = flat_rag_fn(question)
        flat_time = time.time() - start
        flat_answer = flat_result.get("answer", "")

        # GraphRAG
        print("\n[GraphRAG]")
        start = time.time()
        graph_result = graph_rag_fn(G, question)
        graph_time = time.time() - start
        graph_answer = graph_result.get("answer", "")

        # Judge
        print("\n[Judging]")
        judge_result = llm_judge(question, flat_answer, graph_answer)

        result = {
            "question": question,
            "flat_answer": flat_answer[:500],
            "graph_answer": graph_answer[:500],
            "flat_time": round(flat_time, 2),
            "graph_time": round(graph_time, 2),
            "flat_entity": "",
            "graph_entity": graph_result.get("entity", ""),
            "graph_subgraph_size": graph_result.get("subgraph_size", 0),
            "judge": judge_result,
        }
        results.append(result)

        # Print comparison
        print(f"\n  Winner: {judge_result.get('winner', 'unknown')}")
        print(f"  Score A (Flat): {judge_result.get('score_a', '?')} / Score B (Graph): {judge_result.get('score_b', '?')}")
        print(f"  Reason: {judge_result.get('reason', '')[:120]}")
        print(f"  Times: Flat={flat_time:.1f}s, Graph={graph_time:.1f}s")

    return results


def print_comparison_table(results: list[dict]):
    """Print a formatted comparison table."""
    headers = ["#", "Question (truncated)", "Winner", "Flat Score", "Graph Score", "Flat Time", "Graph Time"]
    rows = []

    for i, r in enumerate(results):
        q_short = r["question"][:45] + "..." if len(r["question"]) > 45 else r["question"]
        judge = r.get("judge", {})
        winner = judge.get("winner", "?")
        score_a = judge.get("score_a", "?")
        score_b = judge.get("score_b", "?")
        rows.append([i+1, q_short, winner, score_a, score_b, f"{r['flat_time']:.1f}s", f"{r['graph_time']:.1f}s"])

    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))

    # Summary stats
    flat_wins = sum(1 for r in results if r.get("judge", {}).get("winner") == "A")
    graph_wins = sum(1 for r in results if r.get("judge", {}).get("winner") == "B")
    ties = sum(1 for r in results if r.get("judge", {}).get("winner") == "TIE")

    print(f"\nSummary: Flat RAG wins: {flat_wins}, GraphRAG wins: {graph_wins}, Ties: {ties}")

    avg_flat_time = sum(r["flat_time"] for r in results) / len(results) if results else 0
    avg_graph_time = sum(r["graph_time"] for r in results) / len(results) if results else 0
    print(f"Avg query time: FlatRAG={avg_flat_time:.1f}s, GraphRAG={avg_graph_time:.1f}s")


def save_results(results: list[dict]):
    """Save evaluation results to JSON."""
    ensure_dir(OUTPUT_DIR)
    path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nEvaluation results saved to {path}")
