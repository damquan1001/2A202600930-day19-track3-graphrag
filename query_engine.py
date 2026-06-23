"""
Step 3: GraphRAG Query Engine — Multi-hop graph traversal + LLM answer generation.
"""

import networkx as nx
from collections import deque
from typing import Optional

from utils import llm_call, get_tracker
from config import GRAPH_MAX_HOPS


EXTRACT_ENTITY_PROMPT = """Given the following question, identify the main entity (company, person, technology, concept, etc.) that the question is asking about.

Return ONLY the entity name, nothing else. If there are multiple entities, return the primary one.

Question: {question}

Entity:"""


TEXTUALIZATION_PROMPT = """Below is a knowledge graph subgraph showing entities and their relationships related to the question.

{graph_text}

Based on this information, answer the following question concisely and accurately:

Question: {question}

If the graph information is insufficient to fully answer, say so clearly. Cite specific relationships from the graph where possible."""


def extract_query_entity(question: str) -> Optional[str]:
    """Use LLM to extract the main entity from a user question."""
    try:
        result, pt, ct, cost, duration = llm_call(
            prompt=EXTRACT_ENTITY_PROMPT.format(question=question),
        )
        get_tracker().add_usage("Query: entity extraction", pt, ct, cost, duration)
        entity = result.strip().strip('"').strip("'").strip(".")
        return entity if entity else None
    except Exception as e:
        print(f"  [WARN] Entity extraction failed: {e}")
        return None


def bfs_traverse(
    G: nx.DiGraph,
    start_entity: str,
    max_hops: int = GRAPH_MAX_HOPS,
) -> nx.DiGraph:
    """
    BFS traversal from start_entity up to max_hops.
    Returns a subgraph of the traversed neighborhood.
    """
    from collections import deque

    # Find matching nodes (case-insensitive)
    matched_nodes = []
    entity_lower = start_entity.lower()
    for node in G.nodes():
        if entity_lower in node.lower():
            matched_nodes.append(node)

    if not matched_nodes:
        return None, []

    visited = set()
    queue = deque()
    subgraph_nodes = set()

    for node in matched_nodes:
        queue.append((node, 0))
        visited.add(node)
        subgraph_nodes.add(node)

    while queue:
        current, depth = queue.popleft()
        if depth >= max_hops:
            continue

        # Traverse both directions
        for neighbor in G.successors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                subgraph_nodes.add(neighbor)
                queue.append((neighbor, depth + 1))

        for neighbor in G.predecessors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                subgraph_nodes.add(neighbor)
                queue.append((neighbor, depth + 1))

    subgraph = G.subgraph(subgraph_nodes).copy()
    return subgraph, list(subgraph_nodes)


def textualize_subgraph(subgraph: nx.DiGraph, start_entity: str) -> str:
    """Convert a subgraph into a human-readable text paragraph."""
    if subgraph.number_of_nodes() == 0:
        return "No relevant information found in the knowledge graph."

    lines = []
    for u, v, data in subgraph.edges(data=True):
        relations = data.get("relations", ["RELATED_TO"])
        for rel in relations:
            lines.append(f"- **{u}** --[{rel}]--> **{v}**")

    if not lines:
        return f"Entity '{start_entity}' found in graph but no direct relationships extracted."

    return "\n".join(lines)


def query_graph(
    G: nx.DiGraph,
    question: str,
    max_hops: int = GRAPH_MAX_HOPS,
) -> dict:
    """
    Full GraphRAG query pipeline:
    1. Extract entity from question
    2. BFS traverse graph
    3. Textualize subgraph
    4. LLM answers from textualized context
    """
    result = {
        "question": question,
        "entity": None,
        "subgraph_nodes": [],
        "subgraph_size": 0,
        "graph_context": "",
        "answer": "",
        "error": None,
    }

    # Step 1: Extract entity
    print(f"\n[GraphRAG] Extracting entity from question...")
    entity = extract_query_entity(question)
    result["entity"] = entity
    if not entity:
        result["error"] = "Could not extract entity from question"
        return result

    print(f"  Entity: {entity}")

    # Step 2: BFS Traverse
    print(f"  Traversing graph (max {max_hops}-hop)...")
    subgraph, nodes_found = bfs_traverse(G, entity, max_hops)
    if subgraph is None or subgraph.number_of_nodes() == 0:
        result["error"] = f"Entity '{entity}' not found in knowledge graph"
        return result

    result["subgraph_nodes"] = nodes_found
    result["subgraph_size"] = subgraph.number_of_nodes()
    print(f"  Found {subgraph.number_of_nodes()} related nodes")

    # Step 3: Textualize
    graph_text = textualize_subgraph(subgraph, entity)
    result["graph_context"] = graph_text

    print(f"  Graph context: {len(graph_text)} chars")

    # Step 4: LLM Answer
    print(f"  Generating answer from graph context...")
    try:
        answer, pt, ct, cost, duration = llm_call(
            prompt=TEXTUALIZATION_PROMPT.format(graph_text=graph_text, question=question),
        )
        get_tracker().add_usage("GraphRAG: answer generation", pt, ct, cost, duration)
        result["answer"] = answer
    except Exception as e:
        result["error"] = f"Answer generation failed: {e}"
        print(f"  [ERROR] {e}")

    return result
