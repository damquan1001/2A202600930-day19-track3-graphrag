"""
Step 4: Flat RAG Baseline — TF-IDF vector search + LLM answer.
Lightweight: no large model loading, no ChromaDB overhead.
"""

import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import llm_call, ensure_dir, get_tracker
from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL


RAG_PROMPT = """You are a helpful assistant answering questions about the electric vehicle (EV) industry.

Use the following retrieved context to answer the user's question. If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer concisely and cite relevant parts of the context."""


class FlatRAG:
    """
    Lightweight TF-IDF based RAG. No large model loading.
    """

    def __init__(self, persist_dir: str = None):
        self.vectorizer = None
        self.chunks = []
        self.chunk_matrix = None
        self.persist_dir = persist_dir

    def _chunk_documents(self, documents: list[dict]) -> list[dict]:
        """Split documents into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = []
        for doc in documents:
            text = doc["full_content"]
            if not text or len(text.strip()) < 20:
                continue
            doc_chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(doc_chunks):
                chunks.append({
                    "id": f"{doc['file']}_{i}",
                    "text": chunk,
                    "file": doc["file"],
                    "title": doc["title"],
                    "query": doc["query"],
                })
        return chunks

    def build_index(self, documents: list[dict]):
        """Build TF-IDF index from documents."""
        print("Chunking documents...")
        self.chunks = self._chunk_documents(documents)
        print(f"  Created {len(self.chunks)} chunks")

        print("Building TF-IDF index (lightweight)...")
        texts = [c["text"] for c in self.chunks]

        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            lowercase=True,
        )
        self.chunk_matrix = self.vectorizer.fit_transform(texts)
        print(f"  Index built: {self.chunk_matrix.shape[1]} features")

        # Persist
        if self.persist_dir:
            ensure_dir(self.persist_dir)
            path = os.path.join(self.persist_dir, "flat_rag.pkl")
            with open(path, "wb") as f:
                pickle.dump({
                    "vectorizer": self.vectorizer,
                    "chunks": self.chunks,
                    "matrix": self.chunk_matrix,
                }, f)
            print(f"  Index saved to {path}")

    def load_index(self, persist_dir: str) -> bool:
        """Load previously built index."""
        path = os.path.join(persist_dir, "flat_rag.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.chunks = data["chunks"]
        self.chunk_matrix = data["matrix"]
        print(f"Loaded TF-IDF index: {len(self.chunks)} chunks, {self.chunk_matrix.shape[1]} features")
        return True

    def query(self, question: str, top_k: int = TOP_K_RETRIEVAL) -> dict:
        """
        Query: TF-IDF transform question → cosine similarity → retrieve chunks → LLM answer.
        """
        result = {
            "question": question,
            "retrieved_chunks": [],
            "context": "",
            "answer": "",
            "error": None,
        }

        if self.vectorizer is None or self.chunk_matrix is None:
            result["error"] = "Index not built. Call build_index() first."
            return result

        # Transform question
        q_vec = self.vectorizer.transform([question])

        # Cosine similarity
        scores = cosine_similarity(q_vec, self.chunk_matrix).flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]

        # Build context
        context_parts = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self.chunks[idx]
            context_parts.append(
                f"[Source: {chunk.get('title', 'Unknown')} ({chunk.get('file', '')})]\n{chunk['text']}"
            )
            result["retrieved_chunks"].append({
                "text": chunk["text"][:200],
                "title": chunk["title"],
                "file": chunk["file"],
                "score": float(scores[idx]),
            })

        if not context_parts:
            result["error"] = "No relevant documents found"
            return result

        context = "\n\n---\n\n".join(context_parts)
        result["context"] = context

        # Generate answer
        try:
            answer, pt, ct, cost, duration = llm_call(
                prompt=RAG_PROMPT.format(context=context, question=question),
            )
            get_tracker().add_usage("FlatRAG: answer", pt, ct, cost, duration)
            result["answer"] = answer
        except Exception as e:
            result["error"] = f"Answer generation failed: {e}"

        return result


# Convenience functions matching the original API
def build_vector_store(documents: list[dict], **kwargs) -> FlatRAG:
    """Build and return a FlatRAG instance."""
    rag = FlatRAG(persist_dir=kwargs.get("persist_dir"))
    rag.build_index(documents)
    return rag


def query_flat_rag(rag: FlatRAG, question: str, **kwargs) -> dict:
    """Query a FlatRAG instance."""
    return rag.query(question)
