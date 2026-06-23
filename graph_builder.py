"""
Step 2: Graph Construction — Build NetworkX graph from triples and visualize.
"""

import os
import pickle
import json
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter

from config import GRAPH_DIR, OUTPUT_DIR
from utils import ensure_dir


def build_graph(triples: list[dict]) -> nx.DiGraph:
    """
    Build a directed graph from triples.
    - Nodes: entities
    - Edges: relations (directed)
    """
    G = nx.DiGraph()

    for t in triples:
        subj = t["subject"]
        obj = t["object"]
        rel = t["relation"]

        # Add nodes with attributes
        if not G.has_node(subj):
            G.add_node(subj, type="entity", first_seen=t.get("source_file", ""))
        if not G.has_node(obj):
            G.add_node(obj, type="entity", first_seen=t.get("source_file", ""))

        # Add edge (multiple relations between same nodes allowed as different types)
        if G.has_edge(subj, obj):
            # Append relation type to existing edge
            existing = G[subj][obj].get("relations", [])
            if rel not in existing:
                existing.append(rel)
                G[subj][obj]["relations"] = existing
        else:
            G.add_edge(subj, obj, relations=[rel])

    return G


def save_graph(G: nx.DiGraph, path: str):
    """Save graph as pickle."""
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved to {path}")


def load_graph(path: str) -> nx.DiGraph:
    """Load graph from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def graph_statistics(G: nx.DiGraph) -> dict:
    """Compute basic graph statistics."""
    stats = {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "connected_components": nx.number_weakly_connected_components(G),
        "avg_degree": sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        "is_directed": G.is_directed(),
    }

    # Top entities by degree centrality
    centrality = nx.degree_centrality(G)
    top_entities = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:20]
    stats["top_entities"] = [(name, round(cent, 4)) for name, cent in top_entities]

    # Relation type distribution
    rel_counter = Counter()
    for _, _, data in G.edges(data=True):
        for rel in data.get("relations", []):
            rel_counter[rel] += 1
    stats["relation_distribution"] = rel_counter.most_common(15)

    return stats


def visualize_graph(
    G: nx.DiGraph,
    output_path: str,
    max_nodes: int = 150,
    figsize: tuple = (20, 16),
    node_size_factor: int = 500,
):
    """
    Visualize the graph, showing top entities by connectivity.
    Only shows the top max_nodes to keep visualization readable.
    """
    if G.number_of_nodes() == 0:
        print("Empty graph, nothing to visualize.")
        return

    # Select subgraph: top nodes by degree
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    if len(top_nodes) > max_nodes:
        selected = set(n for n, d in top_nodes[:max_nodes])
        # Also include any neighbors of selected nodes for connectivity
        extra = set()
        for n in selected:
            extra.update(G.predecessors(n))
            extra.update(G.successors(n))
        selected.update(extra)
        G_viz = G.subgraph(selected).copy()
    else:
        G_viz = G.copy()

    if G_viz.number_of_nodes() == 0:
        print("No nodes in visualization subgraph.")
        return

    print(f"Visualizing graph: {G_viz.number_of_nodes()} nodes, {G_viz.number_of_edges()} edges")

    # Layout
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G_viz, k=2, iterations=50, seed=42)

    # Node sizes by degree
    node_degrees = dict(G_viz.degree())
    node_sizes = [max(100, node_degrees[n] * node_size_factor) for n in G_viz.nodes()]

    # Node colors
    node_colors = [
        "#1f77b4"  # Blue for companies
        if any(kw in n.lower() for kw in ["tesla", "ford", "gm", "byd", "nio", "rivian",
                                           "lucid", "volkswagen", "toyota", "honda",
                                           "nissan", "bmw", "mercedes", "hyundai",
                                           "fisker", "polestar", "zeekr", "vinfast",
                                           "nikola", "ree", "arrival", "canoo",
                                           "proterra", "lordstown", "mullen",
                                           "electra", "faraday", "karma"])
        else "#2ca02c"  # Green for others
        for n in G_viz.nodes()
    ]

    # Draw
    nx.draw_networkx_nodes(G_viz, pos, node_size=node_sizes, node_color=node_colors,
                           alpha=0.8, edgecolors="#333")
    nx.draw_networkx_edges(G_viz, pos, alpha=0.3, arrows=True, arrowsize=10,
                           edge_color="#666", connectionstyle="arc3,rad=0.1")

    # Labels for top nodes only
    top_labels = {n: n for n, d in top_nodes[:40] if n in G_viz}
    nx.draw_networkx_labels(G_viz, pos, labels=top_labels, font_size=8,
                            font_weight="bold")

    plt.title(f"EV Industry Knowledge Graph ({G_viz.number_of_nodes()} nodes, "
              f"{G_viz.number_of_edges()} edges)", fontsize=14, pad=20)
    plt.axis("off")
    plt.tight_layout()

    ensure_dir(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph visualization saved to {output_path}")


def run_graph_construction(triples: list[dict]) -> nx.DiGraph:
    """Full graph construction pipeline."""
    print(f"\nBuilding graph from {len(triples)} triples...")
    G = build_graph(triples)

    # Save graph
    ensure_dir(GRAPH_DIR)
    graph_path = os.path.join(GRAPH_DIR, "knowledge_graph.pkl")
    save_graph(G, graph_path)

    # Print statistics
    stats = graph_statistics(G)
    print(f"\nGraph Statistics:")
    print(f"  Nodes: {stats['nodes']}")
    print(f"  Edges: {stats['edges']}")
    print(f"  Density: {stats['density']:.6f}")
    print(f"  Connected components: {stats['connected_components']}")
    print(f"  Avg degree: {stats['avg_degree']:.2f}")
    print(f"\n  Top entities by centrality:")
    for name, cent in stats["top_entities"][:10]:
        print(f"    {name}: {cent:.4f}")

    # Visualize
    viz_path = os.path.join(GRAPH_DIR, "graph_visualization.png")
    visualize_graph(G, viz_path)

    # Save stats as JSON
    stats_path = os.path.join(OUTPUT_DIR, "graph_statistics.json")
    ensure_dir(OUTPUT_DIR)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Statistics saved to {stats_path}")

    return G
