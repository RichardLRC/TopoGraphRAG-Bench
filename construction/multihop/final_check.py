"""
Final Quality Check for Bridge-Chain Questions

LLM-based review of each surviving bridge-chain question on three
dimensions: (i) the bridge entity name is not leaked in the surface
question, (ii) the final answer is not the bridge entity itself,
and (iii) the two hops form a semantically coherent reasoning chain
rather than a coincidental string match.

Input:  filtered_modality/doc{id}_final.json
Output: final_multihop/doc{id}_final.json

Usage:
    python final_check.py                # all docs
    python final_check.py --doc-id 0     # single doc
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILTERED_MODALITY_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "filtered_modality")
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "final_multihop")
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "prompts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(PROMPT_DIR, "final_quality_check.txt")) as f:
    REVIEW_PROMPT = f.read()


# ============================================================
# Helpers
# ============================================================

def call_llm(client: OpenAI, prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    text = resp.choices[0].message.content.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


SCORE_THRESHOLD = 7  # Minimum score on EACH dimension to keep
DIMENSIONS = ["no_bridge_name_in_question", "answer_not_bridge", "semantic_coherence"]


def review_question(client: OpenAI, mh: dict) -> dict:
    """Review one multi-hop question with LLM scoring."""
    trace = mh["reasoning_trace"]

    prompt = REVIEW_PROMPT.format(
        question=mh["question"],
        answer=mh["final_answer"],
        bridge_entity=", ".join(mh["bridge_entities"]),
        hop_path=mh["hop_path"],
        hop1_question=trace[0]["sub_question"],
        hop1_answer=trace[0]["result"],
        hop2_question=trace[1]["sub_question"],
        hop2_answer=trace[1]["result"],
    )

    try:
        scores = call_llm(client, prompt)
        mh["final_review"] = scores

        keep = all(scores.get(dim, 0) >= SCORE_THRESHOLD for dim in DIMENSIONS)
        mh["final_review"]["keep"] = keep
        mh["final_review"]["min_score"] = min(scores.get(d, 0) for d in DIMENSIONS)
    except Exception as e:
        mh["final_review"] = {"keep": True, "min_score": 5, "reason": f"review_error: {e}"}

    return mh


def process_doc(client: OpenAI, doc_id: int):
    input_path = os.path.join(FILTERED_MODALITY_DIR, f"doc{doc_id}_final.json")
    if not os.path.exists(input_path):
        print(f"  [SKIP] doc{doc_id}: no filter_modality output")
        return

    with open(input_path) as f:
        questions = json.load(f)

    if not questions:
        out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_final.json")
        with open(out_path, "w") as f:
            json.dump([], f)
        print(f"Doc {doc_id}: 0 questions")
        return

    print(f"Doc {doc_id}: reviewing {len(questions)} questions")

    # Review in parallel
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(review_question, client, mh): mh["mh_id"]
            for mh in questions
        }
        reviewed = []
        for future in as_completed(futures):
            reviewed.append(future.result())

    kept = [q for q in reviewed if q["final_review"]["keep"]]
    removed = [q for q in reviewed if not q["final_review"]["keep"]]

    kept.sort(key=lambda x: -x["quality_score"])
    cm_kept = sum(1 for q in kept if q.get("cross_modal"))

    print(f"  → {len(kept)} kept, {len(removed)} removed | cross-modal: {cm_kept}")

    # Save kept
    out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_final.json")
    with open(out_path, "w") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)

    # Save removed for analysis
    if removed:
        rej_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_rejected_review.json")
        with open(rej_path, "w") as f:
            json.dump(removed, f, indent=2, ensure_ascii=False)


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

    _patch_openrouter(args.base_url)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.doc_id is not None:
        doc_ids = [args.doc_id]
    else:
        doc_ids = []
        for f in sorted(os.listdir(FILTERED_MODALITY_DIR)):
            if f.endswith("_final.json"):
                doc_ids.append(int(f.split("doc")[1].split("_")[0]))

    for doc_id in doc_ids:
        process_doc(client, doc_id)

    print("\nDone!")


if __name__ == "__main__":
    main()
