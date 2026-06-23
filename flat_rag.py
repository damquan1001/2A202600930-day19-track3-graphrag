"""
Step 4: Flat RAG Baseline — ChromaDB vector search + LLM answer.
"""

import os
import time
from typing import Optional
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import llm_call, get_embeddings, ensure_dir, get_tracker
from config import CHROMA_DIR, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RETRIEVAL


RAG_PROMPT = """You are a helpful assistant answering questions about the electric vehicle (EV) industry.

Use the following retrieved context to answer the user's question. If the context doesn't contain enough information, say so clearly.

Context:
{context}

Question: {question}

Answer concisely and cite relevant parts of the context."""


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into chunks for embedding."""
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
                "metadata": {
                    "file": doc["file"],
                    "title": doc["title"],
                    "query": doc["query"],
                    "chunk_idx": i,
                }
            })

    return chunks


def build_vector_store(
    documents: list[dict],
    persist_dir: str = CHROMA_DIR,
) -> chromadb.Collection:
    """Build ChromaDB vector store from documents."""
    ensure_dir(persist_dir)

    # Chunk documents
    print("Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    # Get embeddings
    print("Loading embedding model...")
    embedder = get_embeddings()

    # Create ChromaDB
    print("Building vector store...")
    client = chromadb.PersistentClient(path=persist_dir)

    # Delete existing collection if present
    try:
        client.delete_collection("ev_docs")
    except:
        pass

    collection = client.create_collection(
        name="ev_docs",
        metadata={"hnsw:space": "cosine"},
    )

    # Batch insert
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["id"] for c in batch]
        metadatas = [c["metadata"] for c in batch]

        # Embed
        embeddings = embedder.encode(texts, show_progress_bar=(i == 0)).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    print(f"  Indexed {len(chunks)} chunks in ChromaDB")
    return collection


def query_flat_rag(
    collection: chromadb.Collection,
    question: str,
    top_k: int = TOP_K_RETRIEVAL,
) -> dict:
    """
    Query Flat RAG: embed question → retrieve chunks → LLM answer.
    """
    result = {
        "question": question,
        "retrieved_chunks": [],
        "context": "",
        "answer": "",
        "error": None,
    }

    # Embed question
    embedder = get_embeddings()
    query_emb = embedder.encode([question])[0].tolist()

    # Retrieve
    retrieved = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
    )

    if not retrieved["documents"] or not retrieved["documents"][0]:
        result["error"] = "No relevant documents found"
        return result

    # Build context
    context_parts = []
    for i, (doc, meta, dist) in enumerate(zip(
        retrieved["documents"][0],
        retrieved["metadatas"][0],
        retrieved["distances"][0],
    )):
        context_parts.append(f"[Source {i+1}: {meta.get('title', 'Unknown')} "
                           f"({meta.get('file', '')})]\n{doc}")
        result["retrieved_chunks"].append({
            "text": doc,
            "metadata": meta,
            "distance": dist,
        })

    context = "\n\n---\n\n".join(context_parts)
    result["context"] = context

    # Generate answer
    print(f"  Generating answer from {len(retrieved['documents'][0])} chunks...")
    try:
        answer, pt, ct, cost, duration = llm_call(
            prompt=RAG_PROMPT.format(context=context, question=question),
        )
        get_tracker().add_usage("FlatRAG: answer generation", pt, ct, cost, duration)
        result["answer"] = answer
    except Exception as e:
        result["error"] = f"Answer generation failed: {e}"

    return result
