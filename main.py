#!/usr/bin/env python3
"""
Lab Day 19: GraphRAG Pipeline — Main Orchestrator

Complete pipeline:
1. Read dataset
2. Extract entity-relation triples (Indexing)
3. Build knowledge graph (Construction)
4. Query using GraphRAG (multi-hop)
5. Query using Flat RAG (baseline)
6. Evaluate and compare

Usage:
    python main.py [--index] [--questions Q1, Q2, ...]

    --index: Force re-indexing (skip cached triples)
    --questions: Custom questions (comma-separated)
"""

import sys
import os
import time
import json
import argparse

# Add current dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import read_dataset, ensure_dir, get_tracker, llm_call
from config import DATASET_DIR, GRAPH_DIR, OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="GraphRAG Lab Day 19")
    parser.add_argument("--index", action="store_true", help="Force re-indexing")
    parser.add_argument("--max-docs", type=int, default=None, help="Limit docs for indexing")
    parser.add_argument("--questions", type=str, default=None, help="Comma-separated questions")
    args = parser.parse_args()

    print("=" * 70)
    print("LAB DAY 19: GRAPHRAG FOR EV INDUSTRY CORPUS")
    print("=" * 70)

    overall_start = time.time()

    # ============================================================
    # 0. API Key Check
    # ============================================================
    from config import OPENAI_API_KEY
    if not OPENAI_API_KEY:
        print("\n[ERROR] No OpenAI API key configured.")
        print("  Set OPENAI_API_KEY in config.py or as environment variable.")
        print("  Alternatively, set OPENAI_BASE_URL for compatible providers.\n")
        sys.exit(1)

    # ============================================================
    # 1. Read Dataset
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 0: Reading Dataset")
    print("=" * 70)
    documents = read_dataset(DATASET_DIR)
    print(f"Loaded {len(documents)} documents")

    # ============================================================
    # 2. Indexing (Triple Extraction)
    # ============================================================
    triples = None
    triples_path = os.path.join(OUTPUT_DIR, "triples.json")
    ensure_dir(OUTPUT_DIR)

    if not args.index and os.path.exists(triples_path):
        print("\n" + "=" * 70)
        print("STEP 1: Loading cached triples")
        print("=" * 70)
        with open(triples_path, "r", encoding="utf-8") as f:
            triples = json.load(f)
        print(f"Loaded {len(triples)} triples from cache")
    else:
        from indexing import run_indexing
        print("\n" + "=" * 70)
        print("STEP 1: Indexing — Entity/Relation Extraction")
        print("=" * 70)
        print("(This step uses LLM and may take a while...)")
        triples = run_indexing(documents, max_docs=args.max_docs)

    # ============================================================
    # 3. Graph Construction
    # ============================================================
    G = None
    graph_path = os.path.join(GRAPH_DIR, "knowledge_graph.pkl")

    if not args.index and os.path.exists(graph_path):
        from graph_builder import load_graph
        print("\n" + "=" * 70)
        print("STEP 2: Loading cached graph")
        print("=" * 70)
        G = load_graph(graph_path)
        print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        from graph_builder import run_graph_construction
        print("\n" + "=" * 70)
        print("STEP 2: Graph Construction")
        print("=" * 70)
        G = run_graph_construction(triples)

    # ============================================================
    # 4. GraphRAG Query Demo
    # ============================================================
    from query_engine import query_graph
    from flat_rag import build_vector_store, query_flat_rag

    print("\n" + "=" * 70)
    print("STEP 3A: GraphRAG Query Demo")
    print("=" * 70)

    demo_questions = [
        "How did Tesla perform compared to Ford and GM in EV sales?",
        "What is the impact of the Inflation Reduction Act on EV adoption?",
        "How are Chinese EV manufacturers competing in the global market?",
    ]

    for q in demo_questions:
        print(f"\n--- Query: {q} ---")
        result = query_graph(G, q)
        if result.get("answer"):
            print(f"\n[GraphRAG Answer]: {result['answer'][:300]}...")
        elif result.get("error"):
            print(f"\n[GraphRAG Error]: {result['error']}")

    # ============================================================
    # 5. Setup Flat RAG Vector Store
    # ============================================================
    print("\n" + "=" * 70)
    print("STEP 3B: Flat RAG — Building Vector Store")
    print("=" * 70)

    chroma_path = os.path.join(os.path.dirname(__file__), "chroma_db")
    import chromadb
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        collection = client.get_collection("ev_docs")
        print("Using existing ChromaDB collection")
    except:
        print("Building new ChromaDB collection...")
        collection = build_vector_store(documents)

    # Demo Flat RAG queries
    print("\n--- Flat RAG Demo ---")
    for q in demo_questions:
        print(f"\n--- Query: {q} ---")
        result = query_flat_rag(collection, q)
        if result.get("answer"):
            print(f"\n[FlatRAG Answer]: {result['answer'][:300]}...")
        elif result.get("error"):
            print(f"\n[FlatRAG Error]: {result['error']}")

    # ============================================================
    # 6. Evaluation
    # ============================================================
    from evaluation import run_evaluation, print_comparison_table, save_results
    from config import BENCHMARK_QUESTIONS

    custom_questions = args.questions.split(",") if args.questions else None
    eval_questions = custom_questions or BENCHMARK_QUESTIONS

    print("\n" + "=" * 70)
    print("STEP 4: Evaluation — Flat RAG vs GraphRAG")
    print("=" * 70)

    flat_rag_fn = lambda q: query_flat_rag(collection, q)
    graph_rag_fn = lambda G, q: query_graph(G, q)

    results = run_evaluation(flat_rag_fn, graph_rag_fn, G, eval_questions)
    print_comparison_table(results)
    save_results(results)

    # ============================================================
    # 7. Final Summary
    # ============================================================
    overall_time = time.time() - overall_start

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(get_tracker().summary())
    print(f"\nTotal pipeline wall time: {overall_time:.1f}s")
    print("\nOutput files:")
    print(f"  - Knowledge graph: {graph_path}")
    print(f"  - Graph visualization: {os.path.join(GRAPH_DIR, 'graph_visualization.png')}")
    print(f"  - Triples: {triples_path}")
    print(f"  - Evaluation results: {os.path.join(OUTPUT_DIR, 'evaluation_results.json')}")


if __name__ == "__main__":
    main()
