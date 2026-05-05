"""GraphRAG: entity-link → 2-hop BFS → textualize → LLM."""
import json
import time
from collections import deque
from openai import OpenAI

from . import config, graph_builder

client = OpenAI(api_key=config.OPENAI_API_KEY)

ENTITY_PROMPT = """Extract the named entities (people, companies, products) mentioned in the question. Return JSON: {"entities": ["..."]}"""


def extract_question_entities(question: str) -> tuple[list[str], dict]:
    response = client.chat.completions.create(
        model=config.ANSWER_MODEL,
        messages=[
            {"role": "system", "content": ENTITY_PROMPT},
            {"role": "user", "content": question},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    return data.get("entities", []), usage


def link_entities(entities: list[str], graph_nodes: list[str]) -> list[str]:
    """Fuzzy match question entities to graph nodes (lowercase substring)."""
    matched = []
    nodes_lower = {n.lower(): n for n in graph_nodes}
    for entity in entities:
        entity_lower = entity.lower().strip()
        if entity_lower in nodes_lower:
            matched.append(nodes_lower[entity_lower])
            continue
        # substring match
        for nl, n in nodes_lower.items():
            if entity_lower in nl or nl in entity_lower:
                matched.append(n)
                break
    return list(dict.fromkeys(matched))


def bfs_subgraph(graph, seeds: list[str], hops: int = 2):
    """BFS k-hop, treating graph as undirected for traversal but keeping directed edges."""
    visited = set(seeds)
    frontier = deque((seed, 0) for seed in seeds)
    while frontier:
        node, depth = frontier.popleft()
        if depth >= hops:
            continue
        neighbors = set(graph.successors(node)) | set(graph.predecessors(node))
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append((neighbor, depth + 1))
    return graph.subgraph(visited).copy()


def textualize(subgraph) -> str:
    lines = []
    for u, v, data in subgraph.edges(data=True):
        rel = data["relation"].replace("_", " ").lower()
        lines.append(f"- {u} {rel} {v}.")
    return "\n".join(lines)


def answer(question: str, hops: int = 2) -> dict:
    graph = graph_builder.load_graph()
    t0 = time.time()
    entities, ent_usage = extract_question_entities(question)
    seeds = link_entities(entities, list(graph.nodes()))

    if not seeds:
        elapsed = time.time() - t0
        return {
            "answer": "I don't know — no matching entity in the knowledge graph.",
            "entities": entities, "seeds": [], "context": "",
            "prompt_tokens": ent_usage["prompt_tokens"],
            "completion_tokens": ent_usage["completion_tokens"],
            "elapsed_sec": elapsed,
        }

    subG = bfs_subgraph(graph, seeds, hops=hops)
    context = textualize(subG)

    prompt = f"""You are answering a question using a knowledge graph. Use ONLY the facts below. If the answer is not present, say "I don't know".

Knowledge graph facts:
{context}

Question: {question}
Answer:"""
    response = client.chat.completions.create(
        model=config.ANSWER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    elapsed = time.time() - t0
    return {
        "answer": response.choices[0].message.content.strip(),
        "entities": entities,
        "seeds": seeds,
        "subgraph_nodes": subG.number_of_nodes(),
        "subgraph_edges": subG.number_of_edges(),
        "context": context,
        "prompt_tokens": ent_usage["prompt_tokens"] + response.usage.prompt_tokens,
        "completion_tokens": ent_usage["completion_tokens"] + response.usage.completion_tokens,
        "elapsed_sec": elapsed,
    }


if __name__ == "__main__":
    r = answer("Who founded the company that owns Instagram?")
    print(r["answer"])
    print("---seeds:", r["seeds"])
