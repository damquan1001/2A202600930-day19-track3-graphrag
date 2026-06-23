"""
Configuration for GraphRAG Lab Day 19.
Loads settings from .env file or environment variables.
"""

import os
from pathlib import Path

# Load .env file manually (no python-dotenv dependency)
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("\"'")
            # Only set if not already an environment variable
            if key not in os.environ:
                os.environ[key] = val

# --- LLM Configuration ---
# Set your OpenAI-compatible API key in .env or as environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# If you use OpenRouter or another provider, set the base URL
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

# Model name
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")

# --- Embedding Configuration ---
# Using TF-IDF instead of sentence-transformers to avoid loading a 400MB model
# Set to "sentence-transformers" if you want the heavier model
EMBEDDING_BACKEND = "tfidf"

# --- Paths ---
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "graphs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# --- Indexing ---
CHUNK_SIZE = 2000        # characters per chunk for large files
CHUNK_OVERLAP = 200

# --- Retrieval ---
TOP_K_RETRIEVAL = 5      # chunks to retrieve for Flat RAG
GRAPH_MAX_HOPS = 2       # BFS traversal depth for GraphRAG

# --- Benchmark Questions ---
BENCHMARK_QUESTIONS = [
    "How has Tesla's market position changed compared to traditional automakers like Ford and GM?",
    "What are the main factors affecting consumer sentiment towards electric vehicles in the US?",
    "How do US government policies and regulations impact the electric vehicle industry?",
    "What is the relationship between charging infrastructure availability and EV adoption rates?",
    "How are Chinese EV manufacturers (like BYD, NIO) affecting the global EV market competition?",
    "What are the financial performance trends among major EV companies in 2023-2024?",
    "How does the availability of lithium and battery technology affect the EV supply chain?",
]
