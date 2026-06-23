"""
Utility functions: LLM client, token tracking, timing, file helpers.
"""

import time
import json
import os
from typing import Optional
from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL

class UsageTracker:
    """Track token usage and timing across the pipeline."""

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.steps = []

    def add_usage(self, step_name: str, prompt_tokens: int, completion_tokens: int,
                  cost: float, duration: float):
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += cost
        self.steps.append({
            "step": step_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost": cost,
            "duration": duration,
        })

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "TOKEN & COST SUMMARY",
            "=" * 60,
            f"{'Step':<30} {'Prompt':>8} {'Comp':>8} {'Cost':>8} {'Time':>8}",
            "-" * 60,
        ]
        for s in self.steps:
            lines.append(
                f"{s['step']:<30} {s['prompt_tokens']:>8,} {s['completion_tokens']:>8,} "
                f"${s['cost']:<6.4f} {s['duration']:<6.1f}s"
            )
        lines.append("-" * 60)
        lines.append(
            f"{'TOTAL':<30} {self.total_prompt_tokens:>8,} {self.total_completion_tokens:>8,} "
            f"${self.total_cost:<6.4f}"
        )
        lines.append(f"\nTotal time: {sum(s['duration'] for s in self.steps):.1f}s")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cost": self.total_cost,
            "steps": self.steps,
        }


# Global tracker
tracker = UsageTracker()


def get_tracker() -> UsageTracker:
    return tracker


# === LLM Client ===

def get_llm():
    """Get an LLM instance. Raises if no API key configured."""
    from langchain_openai import ChatOpenAI

    api_key = OPENAI_API_KEY
    if not api_key:
        raise ValueError(
            "No OpenAI API key found. Set OPENAI_API_KEY in config.py or environment variable."
        )

    kwargs = {
        "model": LLM_MODEL,
        "temperature": 0.1,
        "api_key": api_key,
    }
    if OPENAI_BASE_URL:
        kwargs["base_url"] = OPENAI_BASE_URL

    return ChatOpenAI(**kwargs)


def get_embeddings():
    """Get a TF-IDF vectorizer for embedding (lightweight, no GPU model load)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(max_features=5000, stop_words="english")


def llm_call(prompt: str, system_prompt: str = None, max_retries: int = 2) -> tuple[str, int, int]:
    """
    Call LLM with prompt, return (response_text, prompt_tokens, completion_tokens).
    Uses global tracker for cost estimation.
    """
    llm = get_llm()
    from langchain.schema import HumanMessage, SystemMessage

    messages = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))

    start = time.time()
    response = llm.invoke(messages)
    duration = time.time() - start

    usage = response.response_metadata.get("token_usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Estimate cost (gpt-4o-mini: $0.15/$0.60 per 1M tokens, gpt-4o: $2.50/$10)
    # We'll use a rough average of $2/$8 per 1M as default
    rate_input = 2.0 / 1_000_000
    rate_output = 8.0 / 1_000_000
    cost = (prompt_tokens * rate_input) + (completion_tokens * rate_output)

    return response.content, prompt_tokens, completion_tokens, cost, duration


# === File helpers ===

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_dataset(dataset_dir: str) -> list[dict]:
    """Read all text files from dataset directory."""
    import glob

    files = sorted(
        glob.glob(os.path.join(dataset_dir, "*.txt")),
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
    )
    documents = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        fname = os.path.basename(fpath)

        # Parse metadata
        query = ""
        title = ""
        link = ""
        full_content = content
        lines = content.split("\n", 10)
        for line in lines:
            if line.startswith("Query: "):
                query = line.replace("Query: ", "").strip()
            elif line.startswith("Title: "):
                title = line.replace("Title: ", "").strip()
            elif line.startswith("Link: "):
                link = line.replace("Link: ", "").strip()

        # Full Content starts after "Full Content:" marker
        if "Full Content:" in content:
            idx = content.index("Full Content:")
            full_content = content[idx + len("Full Content:"):].strip()

        documents.append({
            "file": fname,
            "query": query,
            "title": title,
            "link": link,
            "full_content": full_content,
            "raw": content,
        })
    return documents
