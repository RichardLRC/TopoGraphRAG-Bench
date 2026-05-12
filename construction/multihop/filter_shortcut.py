"""
Single-Source Shortcut Filter

For each bridge-chain question, verify that neither hop's source
layout, by itself, is sufficient to answer the question. We feed
each hop's evidence in isolation to an LLM judge and discard any
question that can be answered from a single source -- those are
shortcut-vulnerable and not genuinely multi-hop.

Input:  bridge_chains/doc{id}_multihop.json + loaded_info.json
Output: filtered_shortcut/doc{id}_multihop_filtered.json

Usage:
    python filter_shortcut.py                # all docs
    python filter_shortcut.py --doc-id 0     # single doc
"""

import json, os, base64, argparse, time
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOADED_INFO_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "loaded_info.json")
BRIDGE_CHAINS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "bridge_chains")
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "filtered_shortcut")
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "prompts")
PAGE_IMAGE_DIR = os.environ.get("TOPOGRAPHRAG_PAGE_IMAGE_DIR", os.path.join(os.environ.get("TOPOGRAPHRAG_DATA_ROOT", "data"), "page_images"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(PROMPT_DIR, "single_source_check.txt")) as f:
    CHECK_PROMPT = f.read()


# ============================================================
# Helpers
# ============================================================

def get_layout_content(layout: dict) -> str:
    """Get text representation of a layout."""
    modality = layout["modality"]
    if modality == "text":
        return layout.get("text", "").strip()
    else:
        parts = []
        if layout.get("caption"):
            parts.append(f"Caption: {layout['caption']}")
        if layout.get("topic"):
            parts.append(f"Topic: {layout['topic']}")
        if layout.get("vlm_text"):
            parts.append(f"Description: {layout['vlm_text']}")
        if layout.get("ocr_text", "").strip():
            parts.append(f"Text in image: {layout['ocr_text'].strip()}")
        return "\n".join(parts)


def load_page_image_b64(doc_name: str, page_id: int) -> str | None:
    base_name = doc_name.replace(".pdf", "")
    path = os.path.join(PAGE_IMAGE_DIR, f"{base_name}_{page_id}.jpg")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def call_llm(client: OpenAI, prompt: str, image_b64: str = None) -> dict:
    if image_b64:
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]}]
    else:
        messages = [{"role": "user", "content": prompt}]

    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=512,
    )
    text = resp.choices[0].message.content.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def answer_matches(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer (fuzzy)."""
    if not predicted or not gold:
        return False
    p = predicted.lower().strip().strip('."\'')
    g = gold.lower().strip().strip('."\'')
    # Exact match
    if p == g:
        return True
    # Substring match (either direction)
    if p in g or g in p:
        return True
    # Number match: strip % and compare
    p_num = p.replace("%", "").replace("$", "").replace(",", "").strip()
    g_num = g.replace("%", "").replace("$", "").replace(",", "").strip()
    if p_num and g_num and p_num == g_num:
        return True
    return False


def check_single_source(client: OpenAI, question: str, gold_answer: str,
                        layout: dict, doc_name: str) -> dict:
    """Check if the question can be answered from a single source layout."""
    content = get_layout_content(layout)
    modality = layout["modality"]
    page_id = layout["page_id"]

    # For figure/table: also send the image
    image_b64 = None
    if modality in ("figure", "table"):
        image_b64 = load_page_image_b64(doc_name, page_id)

    prompt = CHECK_PROMPT.format(
        question=question,
        modality=modality,
        page_id=page_id,
        content=content,
    )

    try:
        result = call_llm(client, prompt, image_b64)
        answerable = result.get("answerable", False)
        predicted = result.get("answer")
        reason = result.get("reason", "")

        # Double check: even if LLM says answerable, verify the answer matches
        if answerable and predicted:
            matches = answer_matches(str(predicted), gold_answer)
        else:
            matches = False

        return {
            "answerable": answerable,
            "answer_matches": matches,
            "predicted_answer": predicted,
            "reason": reason,
        }
    except Exception as e:
        return {"answerable": False, "answer_matches": False, "predicted_answer": None, "reason": str(e)}


def validate_question(client: OpenAI, mh: dict, layouts_map: dict, doc_name: str) -> dict:
    """Validate one multi-hop question with single-source checks."""
    question = mh["question"]
    gold_answer = mh["final_answer"]
    trace = mh["reasoning_trace"]

    hop1_layout = layouts_map.get(trace[0]["source_layout"])
    hop2_layout = layouts_map.get(trace[1]["source_layout"])

    if not hop1_layout or not hop2_layout:
        mh["filter_result"] = "skip_missing_layout"
        return mh

    # Check 1: Can Hop1 source alone answer the full question?
    hop1_check = check_single_source(client, question, gold_answer, hop1_layout, doc_name)

    # Check 2: Can Hop2 source alone answer the full question?
    hop2_check = check_single_source(client, question, gold_answer, hop2_layout, doc_name)

    mh["single_source_checks"] = {
        "hop1_only": hop1_check,
        "hop2_only": hop2_check,
    }

    # Determine filter result
    if hop1_check["answer_matches"]:
        mh["filter_result"] = "fail_hop1_sufficient"
    elif hop2_check["answer_matches"]:
        mh["filter_result"] = "fail_hop2_sufficient"
    else:
        mh["filter_result"] = "pass"

    return mh


def process_doc(client: OpenAI, doc: dict):
    """Process one document."""
    doc_id = doc["doc_id"]
    doc_name = doc["doc_name"]

    mh_path = os.path.join(BRIDGE_CHAINS_DIR, f"doc{doc_id}_multihop.json")
    if not os.path.exists(mh_path):
        print(f"  [SKIP] {mh_path} not found")
        return

    with open(mh_path) as f:
        multihops = json.load(f)

    layouts_map = {l["layout_id"]: l for l in doc["layouts"]}

    print(f"Doc {doc_id}: validating {len(multihops)} multi-hop questions")

    # Validate in parallel
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(validate_question, client, mh, layouts_map, doc_name): mh["mh_id"]
            for mh in multihops
        }
        results = []
        for future in as_completed(futures):
            results.append(future.result())

    # Separate pass/fail
    passed = [r for r in results if r["filter_result"] == "pass"]
    fail_hop1 = [r for r in results if r["filter_result"] == "fail_hop1_sufficient"]
    fail_hop2 = [r for r in results if r["filter_result"] == "fail_hop2_sufficient"]

    print(f"  → {len(passed)} passed, {len(fail_hop1)} failed (hop1 sufficient), "
          f"{len(fail_hop2)} failed (hop2 sufficient)")

    # Sort passed by score
    passed.sort(key=lambda x: -x["quality_score"])

    # Deduplicate: same bridge entity + same final answer = duplicate
    seen = set()
    deduped = []
    for q in passed:
        key = (tuple(q["bridge_entities"]), q["final_answer"].lower().strip())
        if key not in seen:
            seen.add(key)
            deduped.append(q)

    if len(deduped) < len(passed):
        print(f"  → Deduped: {len(passed)} → {len(deduped)}")
    passed = deduped

    out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_multihop_filtered.json")
    with open(out_path, "w") as f:
        json.dump(passed, f, indent=2, ensure_ascii=False)

    # Also save rejected for analysis
    rejected = fail_hop1 + fail_hop2
    if rejected:
        rej_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_rejected.json")
        with open(rej_path, "w") as f:
            json.dump(rejected, f, indent=2, ensure_ascii=False)


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

    with open(LOADED_INFO_PATH) as f:
        data = json.load(f)

    _patch_openrouter(args.base_url)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.doc_id is not None:
        docs = [d for d in data if d["doc_id"] == args.doc_id]
    else:
        docs = data

    for doc in docs:
        process_doc(client, doc)

    print("\nDone!")


if __name__ == "__main__":
    main()
