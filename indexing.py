"""
Step 1: Indexing Pipeline — Extract entity-relation triples from documents using LLM.
"""

import time
import json
import re
from typing import Optional
from utils import llm_call, get_tracker
from config import CHUNK_SIZE, CHUNK_OVERLAP


EXTRACTION_SYSTEM_PROMPT = """You are an expert at extracting structured knowledge from text.
Extract entity-relation triples from the given text about the electric vehicle (EV) industry.

Entities can be:
- Companies (Tesla, BYD, Ford, GM, Rivian, etc.)
- People (Elon Musk, Sam Altman, etc.)
- Products (Model 3, Cybertruck, etc.)
- Technologies (Lithium-ion battery, Solid-state battery, etc.)
- Locations (United States, China, California, etc.)
- Policies/Regulations (IRA, ZEV mandate, etc.)
- Concepts/Events (EV adoption, Charging infrastructure, etc.)
- Organizations (EPA, IEA, DOE, etc.)

Relations should use UPPER_SNAKE_CASE, examples:
FOUNDED_BY, INVESTED_IN, PARTNERED_WITH, COMPETES_WITH, LOCATED_IN,
PRODUCES, DEVELOPED, REGULATES, INCREASES, DECREASES, PROJECTS,
CAUSES, INFLUENCES, PART_OF, HAS_GOAL, LEADS_TO, ACQUIRED, SUBSIDIARY_OF

Return ONLY a JSON array of triples. Each triple must have exactly:
{"subject": "entity1", "relation": "RELATION_NAME", "object": "entity2"}

If the text has no meaningful entities, return an empty array [].

Do NOT include any explanation or markdown formatting outside the JSON array."""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Try to break at a sentence boundary
        if end < len(text):
            # Find last sentence end within the chunk
            for sep in ["\n\n", ". ", ".\n", "! ", "? "]:
                last_break = text.rfind(sep, start, end)
                if last_break > start + chunk_size // 2:
                    end = last_break + len(sep)
                    break
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def extract_triples_from_doc(
    doc: dict,
    chunk_size: int = CHUNK_SIZE,
) -> list[dict]:
    """
    Extract triples from a document using LLM.
    Returns list of {"subject": ..., "relation": ..., "object": ...}.
    """
    text = doc["full_content"]
    if not text or len(text.strip()) < 20:
        return []

    chunks = chunk_text(text, chunk_size)
    all_triples = []

    for i, chunk in enumerate(chunks):
        if len(chunk.strip()) < 20:
            continue

        try:
            result, pt, ct, cost, duration = llm_call(
                prompt=f"Extract entity-relation triples from this text:\n\n{chunk}",
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
            )
            get_tracker().add_usage(
                f"Extract: {doc['file']}:chunk{i}",
                pt, ct, cost, duration
            )

            # Parse JSON from response
            triples = parse_triple_response(result)
            for t in triples:
                t["source_file"] = doc["file"]
                t["source_title"] = doc["title"]
                t["chunk_idx"] = i
            all_triples.extend(triples)

        except Exception as e:
            print(f"  [WARN] Failed to extract triples from {doc['file']}:chunk{i}: {e}")
            continue

    return all_triples


def parse_triple_response(response: str) -> list[dict]:
    """Parse LLM response to extract triple JSON array."""
    # Try to find JSON array in response
    # Look for content between square brackets
    response = response.strip()

    # Remove markdown code fences
    response = re.sub(r"```(?:json)?\s*", "", response)

    # Find the first [ and last ]
    start = response.find("[")
    end = response.rfind("]")
    if start >= 0 and end > start:
        json_str = response[start:end+1]
    else:
        return []

    try:
        triples = json.loads(json_str)
        if not isinstance(triples, list):
            return []
        # Validate triple format
        valid = []
        for t in triples:
            if isinstance(t, dict) and "subject" in t and "relation" in t and "object" in t:
                valid.append({
                    "subject": str(t["subject"]).strip(),
                    "relation": str(t["relation"]).strip().upper(),
                    "object": str(t["object"]).strip(),
                })
        return valid
    except json.JSONDecodeError:
        return []


def normalize_entity(name: str) -> str:
    """Normalize entity name for deduplication."""
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    lower = name.lower()
    # Common normalization
    known_aliases = {
        "elon musk": "Elon Musk",
        "tesla inc": "Tesla",
        "tesla motors": "Tesla",
        "tesla, inc.": "Tesla",
        "general motors": "GM",
        "general motors company": "GM",
        "gm": "GM",
        "ford motor company": "Ford",
        "ford motors": "Ford",
        "ford": "Ford",
        "byd company limited": "BYD",
        "byd co ltd": "BYD",
        "byd": "BYD",
        "nio inc": "NIO",
        "nio": "NIO",
        "rivian automotive": "Rivian",
        "rivian": "Rivian",
        "lucid motors": "Lucid",
        "lucid group": "Lucid",
        "lucid": "Lucid",
        "volkswagen group": "Volkswagen",
        "volkswagen ag": "Volkswagen",
        "vw": "Volkswagen",
        "united states": "United States",
        "united states of america": "United States",
        "us": "United States",
        "u.s.": "United States",
        "china": "China",
        "people's republic of china": "China",
        "european union": "EU",
        "eu": "EU",
        "inflation reduction act": "IRA",
        "inflation reduction act (ira)": "IRA",
    }
    return known_aliases.get(lower, name)


def deduplicate_triples(triples: list[dict]) -> list[dict]:
    """Deduplicate triples by normalizing entity names."""
    seen = set()
    deduped = []
    for t in triples:
        subj = normalize_entity(t["subject"])
        obj = normalize_entity(t["object"])
        rel = t["relation"]
        key = (subj, rel, obj)
        if key not in seen:
            seen.add(key)
            deduped.append({
                "subject": subj,
                "relation": rel,
                "object": obj,
                "source_file": t.get("source_file", ""),
                "source_title": t.get("source_title", ""),
            })
    return deduped


def run_indexing(documents: list[dict], max_docs: Optional[int] = None) -> list[dict]:
    """
    Run the full indexing pipeline on all documents.
    Returns deduplicated triples.
    """
    from tqdm import tqdm

    docs_to_process = documents[:max_docs] if max_docs else documents
    all_triples = []

    print(f"Extracting triples from {len(docs_to_process)} documents...")
    for doc in tqdm(docs_to_process):
        triples = extract_triples_from_doc(doc)
        all_triples.extend(triples)

    print(f"Raw triples extracted: {len(all_triples)}")
    deduped = deduplicate_triples(all_triples)
    print(f"After deduplication: {len(deduped)} unique triples")

    # Save triples
    import json
    import os
    from config import OUTPUT_DIR
    from utils import ensure_dir
    ensure_dir(OUTPUT_DIR)

    out_path = os.path.join(OUTPUT_DIR, "triples.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    print(f"Triples saved to {out_path}")

    return deduped
