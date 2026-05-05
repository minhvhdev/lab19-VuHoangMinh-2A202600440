"""Trích xuất triples (subject, relation, object) từ corpus bằng LLM."""
import json
import re
import time
from openai import OpenAI
from tqdm import tqdm

from . import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

ALIAS_MAP = {
    "facebook": "Meta Platforms",
    "facebook, inc.": "Meta Platforms",
    "thefacebook, inc.": "Meta Platforms",
    "meta": "Meta Platforms",
    "google llc": "Google",
    "alphabet": "Alphabet Inc.",
    "alphabet inc": "Alphabet Inc.",
    "apple computer, inc.": "Apple Inc.",
    "apple": "Apple Inc.",
    "microsoft corporation": "Microsoft",
    "nvidia corporation": "Nvidia",
    "openai global, llc": "OpenAI",
    "tesla": "Tesla, Inc.",
    "tesla inc": "Tesla, Inc.",
    "amazon.com, inc.": "Amazon",
    "amazon web services": "AWS",
    "twitter": "X (Twitter)",
    "x": "X (Twitter)",
    "google deepmind": "DeepMind",
}


def canonicalize(entity: str) -> str:
    e = entity.strip()
    key = e.lower().rstrip(".").strip()
    if key in ALIAS_MAP:
        return ALIAS_MAP[key]
    # extract year
    match = re.fullmatch(r"(?:in\s+)?(\d{4})", key)
    if match:
        return match.group(1)
    return e


def deduplicate(triples: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for triple in triples:
        subject = canonicalize(triple["subject"])
        relation = triple["relation"].strip().upper().replace(" ", "_")
        obj = canonicalize(triple["object"])
        key = (subject.lower(), relation, obj.lower())
        if key in seen or not subject or not obj:
            continue
        seen.add(key)
        result.append({"subject": subject, "relation": relation, "object": obj})
    return result


SYSTEM_PROMPT = """You are an information extraction engine. Given a passage about technology companies, extract factual triples in the form (subject, relation, object).

Rules:
- Subject and object must be named entities (people, companies, products, places, dates).
- Relation should be a SHORT UPPER_SNAKE_CASE verb-phrase (e.g., FOUNDED_BY, ACQUIRED, CEO_OF, FOUNDED_IN, HEADQUARTERED_IN, OWNS, INVESTED_IN, DEVELOPED, RELEASED_IN, SUBSIDIARY_OF).
- Use canonical full names (e.g., "Meta Platforms" not "Meta", "OpenAI" not "the company"). For people, use "First Last".
- Years should be 4-digit strings.
- Extract ALL meaningful facts. One sentence may produce multiple triples.
- Do NOT invent facts not stated in the text.

Return ONLY valid JSON in this exact format:
{"triples": [{"subject": "...", "relation": "...", "object": "..."}]}"""


def extract_triples_from_text(text: str, model: str = None) -> tuple[list[dict], dict]:
    model = model or config.EXTRACT_MODEL
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Passage:\n{text}\n\nExtract triples as JSON."},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }
    return data.get("triples", []), usage


def run(corpus_path=None, output_path=None):
    corpus_path = corpus_path or config.CORPUS_PATH
    output_path = output_path or config.TRIPLES_PATH

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)

    all_triples = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    t0 = time.time()
    per_doc_stats = []

    for doc in tqdm(corpus, desc="Extracting"):
        triples, usage = extract_triples_from_text(doc["text"])
        for triple in triples:
            triple["source_doc"] = doc["id"]
        all_triples.extend(triples)
        total_usage["prompt_tokens"] += usage["prompt_tokens"]
        total_usage["completion_tokens"] += usage["completion_tokens"]
        per_doc_stats.append({"doc": doc["id"], "n_triples": len(triples)})

    deduped = deduplicate(all_triples)
    elapsed = time.time() - t0

    out = {
        "triples": deduped,
        "raw_count": len(all_triples),
        "deduped_count": len(deduped),
        "usage": total_usage,
        "elapsed_sec": elapsed,
        "per_doc": per_doc_stats,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nExtracted {len(all_triples)} raw -> {len(deduped)} deduped triples")
    print(f"Tokens: prompt={total_usage['prompt_tokens']}  completion={total_usage['completion_tokens']}")
    print(f"Time: {elapsed:.1f}s -> saved to {output_path}")
    return out


if __name__ == "__main__":
    run()
