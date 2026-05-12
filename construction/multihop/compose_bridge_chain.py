"""
Bridge-Chain Multi-Hop Question Composition

Compose two-hop bridge-chain questions by pairing a bridge atomic
QA (answer = shared entity E) with a target atomic QA (mentions E)
from different layouts. The LLM rewrites the target question so
that the entity mention is replaced by an indirect description
derived from the bridge question, making the shared entity an
explicit dependency between the two hops.

Input:  atomic_qa/doc{id}_single_hop.json
Output: bridge_chains/doc{id}_multihop.json

Usage:
    python compose_bridge_chain.py                # all docs
    python compose_bridge_chain.py --doc-id 0     # single doc
"""

import json, os, argparse, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI


def _patch_openrouter(base_url):
    """If base_url points at OpenRouter, disable Qwen thinking and pin provider away from Alibaba."""
    if "openrouter" not in base_url.lower():
        return
    from openai.resources.chat.completions import Completions
    _orig = Completions.create
    def _patched(self, *args, **kwargs):
        extra = kwargs.get("extra_body") or {}
        extra.setdefault("reasoning", {"enabled": False})
        extra.setdefault("provider", {
            "order": ["Parasail", "Venice"],
            "ignore": ["Alibaba"],
            "allow_fallbacks": True,
        })
        kwargs["extra_body"] = extra
        return _orig(self, *args, **kwargs)
    Completions.create = _patched


# ============================================================
# Config
# ============================================================
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://localhost:8010/v1")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "EMPTY")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3.5-35b")
NUM_WORKERS = 16
MAX_RETRIES = 2
SCORE_THRESHOLD = 8.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ATOMIC_QA_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "atomic_qa")
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "bridge_chains")
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "prompts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(PROMPT_DIR, "multihop_compose.txt")) as f:
    COMPOSE_PROMPT = f.read()
with open(os.path.join(PROMPT_DIR, "multihop_score.txt")) as f:
    SCORE_PROMPT = f.read()


# ============================================================
# Helpers
# ============================================================

def call_llm(client: OpenAI, prompt: str, max_tokens: int = 1024) -> str:
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def parse_json(text: str) -> dict:
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def find_composable_pairs(single_hops: list[dict]) -> list[dict]:
    """Find all (bridge, target) pairs on the same shared entity."""
    # Group by shared entity
    by_entity = {}
    for q in single_hops:
        ent = q["shared_entity"]
        by_entity.setdefault(ent, {"bridge": [], "target": []})
        by_entity[ent][q["question_role"]].append(q)

    pairs = []
    for entity, groups in by_entity.items():
        for b in groups["bridge"]:
            for t in groups["target"]:
                # Must be different layouts
                if b["source"]["layout_id"] == t["source"]["layout_id"]:
                    continue

                hop_path = f"{b['source']['modality']}→{t['source']['modality']}"
                cross_modal = b["source"]["modality"] != t["source"]["modality"]
                cross_page = b["source"]["page_id"] != t["source"]["page_id"]

                # Skip if target is from figure/table but answer also in text
                if t.get("text_redundant", False) and cross_modal:
                    continue

                pairs.append({
                    "bridge": b,
                    "target": t,
                    "entity": entity,
                    "hop_path": hop_path,
                    "cross_modal": cross_modal,
                    "cross_page": cross_page,
                })

    # Sort: cross-modal first, then cross-page
    pairs.sort(key=lambda x: (-x["cross_modal"], -x["cross_page"]))
    return pairs


def compose_one(client: OpenAI, pair: dict, doc_id: int, mh_id: str) -> dict | None:
    """Compose one multi-hop question from a (bridge, target) pair."""
    b, t = pair["bridge"], pair["target"]

    # Step 4a: Compose
    prompt = COMPOSE_PROMPT.format(
        hop1_modality=b["source"]["modality"],
        hop1_page=b["source"]["page_id"],
        hop1_question=b["question"],
        hop1_answer=b["answer"],
        hop2_modality=t["source"]["modality"],
        hop2_page=t["source"]["page_id"],
        hop2_question=t["question"],
        hop2_answer=t["answer"],
        bridge_entity=pair["entity"],
    )

    try:
        result = parse_json(call_llm(client, prompt, max_tokens=512))
        composed_q = result.get("composed_question", "")
        final_answer = result.get("final_answer", t["answer"])
        indirect_desc = result.get("indirect_description", "")
    except Exception as e:
        return None

    if not composed_q or len(composed_q) < 15:
        return None

    # Check no leakage: bridge entity should not appear in composed question
    entity_lower = pair["entity"].lower()
    if entity_lower in composed_q.lower():
        return None  # Leakage

    # Step 4b: Score
    score_prompt = SCORE_PROMPT.format(
        question=composed_q,
        answer=final_answer,
        bridge_entity=pair["entity"],
        hop_path=pair["hop_path"],
        hop1_modality=b["source"]["modality"],
        hop1_question=b["question"],
        hop1_answer=b["answer"],
        hop2_modality=t["source"]["modality"],
        hop2_question=t["question"],
        hop2_answer=t["answer"],
    )

    try:
        score_result = parse_json(call_llm(client, score_prompt, max_tokens=256))
        score = float(score_result.get("score", 0))
    except Exception:
        score = 5.0

    if score < SCORE_THRESHOLD:
        return None

    return {
        "mh_id": mh_id,
        "doc_id": doc_id,
        "num_hops": 2,
        "question": composed_q,
        "final_answer": final_answer,
        "answer_type": t["answer_type"],
        "bridge_entities": [pair["entity"]],
        "indirect_description": indirect_desc,
        "hop_path": pair["hop_path"],
        "cross_modal": pair["cross_modal"],
        "cross_page": pair["cross_page"],
        "quality_score": score,
        "reasoning_trace": [
            {
                "hop": 1,
                "source_modality": b["source"]["modality"],
                "source_page": b["source"]["page_id"],
                "source_layout": b["source"]["layout_id"],
                "sub_question": b["question"],
                "result": b["answer"],
            },
            {
                "hop": 2,
                "source_modality": t["source"]["modality"],
                "source_page": t["source"]["page_id"],
                "source_layout": t["source"]["layout_id"],
                "sub_question": t["question"],
                "result": t["answer"],
            },
        ],
        "evidence_sources": [
            {"modality": b["source"]["modality"], "layout_id": b["source"]["layout_id"], "page_id": b["source"]["page_id"]},
            {"modality": t["source"]["modality"], "layout_id": t["source"]["layout_id"], "page_id": t["source"]["page_id"]},
        ],
        "source_single_hops": [b["sh_id"], t["sh_id"]],
    }


def process_doc(client: OpenAI, doc_id: int):
    """Process one document."""
    sh_path = os.path.join(ATOMIC_QA_DIR, f"doc{doc_id}_single_hop.json")
    if not os.path.exists(sh_path):
        print(f"  [SKIP] {sh_path} not found")
        return

    with open(sh_path) as f:
        single_hops = json.load(f)

    pairs = find_composable_pairs(single_hops)
    cross_modal_pairs = [p for p in pairs if p["cross_modal"]]

    print(f"Doc {doc_id}: {len(single_hops)} single-hops → {len(pairs)} pairs "
          f"({len(cross_modal_pairs)} cross-modal)")

    if not pairs:
        return

    # Compose in parallel
    results = []
    mh_counter = [0]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        for pair in pairs:
            mh_counter[0] += 1
            mh_id = f"doc{doc_id}_mh_{mh_counter[0]:04d}"
            future = executor.submit(compose_one, client, pair, doc_id, mh_id)
            futures[future] = pair

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    # Deduplicate: same pair of source layouts should not appear twice
    seen_pairs = set()
    deduped = []
    for r in results:
        key = tuple(sorted([r["evidence_sources"][0]["layout_id"],
                            r["evidence_sources"][1]["layout_id"]]))
        if key not in seen_pairs:
            seen_pairs.add(key)
            deduped.append(r)

    # Sort by quality score descending
    deduped.sort(key=lambda x: -x["quality_score"])

    cross_modal = sum(1 for r in deduped if r["cross_modal"])
    print(f"  → {len(deduped)} multi-hop questions ({cross_modal} cross-modal)")

    out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_multihop.json")
    with open(out_path, "w") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", type=int, default=None)
    parser.add_argument("--base-url", default=VLM_BASE_URL)
    parser.add_argument("--api-key", default=VLM_API_KEY,
                        help="EMPTY for local vLLM; real key for OpenRouter")
    parser.add_argument("--model", default=VLM_MODEL)
    args = parser.parse_args()

    # Determine doc ids
    if args.doc_id is not None:
        doc_ids = [args.doc_id]
    else:
        doc_ids = []
        for f in sorted(os.listdir(ATOMIC_QA_DIR)):
            if f.endswith("_single_hop.json"):
                doc_ids.append(int(f.split("doc")[1].split("_")[0]))

    print(f"Processing {len(doc_ids)} docs")
    _patch_openrouter(args.base_url)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    for doc_id in doc_ids:
        process_doc(client, doc_id)

    print("\nDone!")


if __name__ == "__main__":
    main()
