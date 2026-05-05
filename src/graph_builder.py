"""Xây dựng đồ thị NetworkX (+ Neo4j optional) từ triples."""
import json
import pickle
import networkx as nx
import matplotlib.pyplot as plt

from . import config


def build_networkx(triples: list[dict]) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for triple in triples:
        subject, relation, obj = triple["subject"], triple["relation"], triple["object"]
        graph.add_node(subject)
        graph.add_node(obj)
        graph.add_edge(subject, obj, relation=relation, source_doc=triple.get("source_doc"))
    return graph


def save_graph(graph: nx.MultiDiGraph, path=None):
    path = path or config.GRAPH_PATH
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_graph(path=None) -> nx.MultiDiGraph:
    path = path or config.GRAPH_PATH
    with open(path, "rb") as f:
        return pickle.load(f)


def visualize(graph: nx.MultiDiGraph, path=None, max_nodes=60):
    path = path or config.GRAPH_PNG
    if graph.number_of_nodes() > max_nodes:
        degree_dict = dict(graph.degree())
        top = sorted(degree_dict, key=degree_dict.get, reverse=True)[:max_nodes]
        subgraph = graph.subgraph(top).copy()
    else:
        subgraph = graph

    plt.figure(figsize=(20, 14))
    pos = nx.spring_layout(subgraph, k=1.2, iterations=80, seed=42)
    nx.draw_networkx_nodes(subgraph, pos, node_size=700, node_color="#90caf9", alpha=0.9)
    nx.draw_networkx_labels(subgraph, pos, font_size=8)
    nx.draw_networkx_edges(subgraph, pos, edge_color="#888", arrows=True, alpha=0.5,
                           connectionstyle="arc3,rad=0.1")
    edge_labels = {(u, v): d["relation"] for u, v, d in subgraph.edges(data=True)}
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=6)
    plt.title(f"Tech Company Knowledge Graph ({subgraph.number_of_nodes()} nodes / {subgraph.number_of_edges()} edges)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Graph image -> {path}")


def push_to_neo4j(triples: list[dict]):
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("neo4j driver not installed; skipping")
        return
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        for triple in triples:
            session.run(
                "MERGE (a:Entity {name:$s}) MERGE (b:Entity {name:$o}) "
                "MERGE (a)-[r:REL {type:$r}]->(b)",
                s=triple["subject"], o=triple["object"], r=triple["relation"],
            )
    driver.close()
    print(f"Pushed {len(triples)} triples to Neo4j @ {config.NEO4J_URI}")


def run(triples_path=None, push_neo4j: bool = False):
    triples_path = triples_path or config.TRIPLES_PATH
    with open(triples_path, encoding="utf-8") as f:
        data = json.load(f)
    triples = data["triples"]

    graph = build_networkx(triples)
    save_graph(graph)
    visualize(graph)
    print(f"Graph: {graph.number_of_nodes()} nodes / {graph.number_of_edges()} edges")

    if push_neo4j:
        push_to_neo4j(triples)
    return graph


if __name__ == "__main__":
    import sys
    run(push_neo4j="--neo4j" in sys.argv)
