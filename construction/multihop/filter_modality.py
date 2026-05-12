"""
Modality Necessity Filter (OCR-aware)

For each cross-modal question, verify that it cannot be answered
from a single modality alone. The text-only check feeds text-modality
content together with OCR'd figure/table content -- matching what
text-only graph RAG systems (LightRAG, HippoRAG, MS-GraphRAG)
actually receive when ingesting text_chunks_original -- so the
cross-modal label is not based on layout type alone.

Caption and VLM-generated descriptions are intentionally excluded:
they are not part of the ingestion stream of any deployed text-only
graph RAG system, and including them would simulate a baseline that
does not exist.

Input:  filtered_shortcut/doc{id}_multihop_filtered.json + loaded_info.json
Output: filtered_modality/doc{id}_final.json
        filtered_modality/doc{id}_rejected_modality.json

Usage:
    python filter_modality.py                # all docs
    python filter_modality.py --doc-id 0     # single doc
    python filter_modality.py \
        --base-url https://openrouter.ai/api/v1 \
        --model qwen/qwen3.5-35b-a3b \
        --api-key <api-key>
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
# Defaults
# ============================================================
DEFAULT_BASE_URL = "http://localhost:8010/v1"
DEFAULT_API_KEY = os.environ.get("VLM_API_KEY") or os.environ.get("OPENROUTER_KEY") or "EMPTY"
DEFAULT_MODEL = "qwen3.5-35b"
NUM_WORKERS = 16

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOADED_INFO_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "loaded_info.json")
FILTERED_SHORTCUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "filtered_shortcut")
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "filtered_modality")    # ← v2 output folder
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "prompts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PAGE_IMAGE_DIR = os.environ.get("TOPOGRAPHRAG_PAGE_IMAGE_DIR", os.path.join(os.environ.get("TOPOGRAPHRAG_DATA_ROOT", "data"), "page_images"))

with open(os.path.join(PROMPT_DIR, "modality_check.txt")) as f:
    TEXT_ONLY_PROMPT = f.read()
with open(os.path.join(PROMPT_DIR, "visual_only_check.txt")) as f:
    VISUAL_ONLY_PROMPT = f.read()


# ============================================================
# Helpers (unchanged from v1)
# ============================================================

def call_llm(client: OpenAI, prompt: str, model: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
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
    if not predicted or not gold:
        return False
    p = predicted.lower().strip().strip('."\'')
    g = gold.lower().strip().strip('."\'')
    if p == g:
        return True
    if p in g or g in p:
        return True
    p_num = p.replace("%", "").replace("$", "").replace(",", "").strip()
    g_num = g.replace("%", "").replace("$", "").replace(",", "").strip()
    if p_num and g_num and p_num == g_num:
        return True
    return False


def load_page_image_b64(doc_name: str, page_id: int) -> str | None:
    base_name = doc_name.replace(".pdf", "")
    path = os.path.join(PAGE_IMAGE_DIR, f"{base_name}_{page_id}.jpg")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def call_llm_with_images(client: OpenAI, prompt: str, images_b64: list[str], model: str) -> dict:
    """Call LLM with multiple images + text prompt."""
    content = []
    for img in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    content.append({"type": "text", "text": prompt})

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
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


def check_visual_only(client: OpenAI, question: str, gold_answer: str,
                      visual_page_images: list[str], doc_name: str, model: str) -> dict:
    """Check if the question can be answered using only figure/table images."""
    images = []
    for page_id in visual_page_images:
        img = load_page_image_b64(doc_name, page_id)
        if img:
            images.append(img)
        if len(images) >= 5:
            break

    if not images:
        return {"answerable": False, "answer_matches": False, "predicted_answer": None, "reason": "no images"}

    prompt = VISUAL_ONLY_PROMPT.format(question=question)

    try:
        result = call_llm_with_images(client, prompt, images, model)
        answerable = result.get("answerable", False)
        predicted = result.get("answer")
        matches = answer_matches(str(predicted), gold_answer) if answerable and predicted else False
        return {
            "answerable": answerable,
            "answer_matches": matches,
            "predicted_answer": predicted,
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        return {"answerable": False, "answer_matches": False, "predicted_answer": None, "reason": str(e)}


def check_text_only(client: OpenAI, question: str, gold_answer: str,
                    all_text_contents: list[str], model: str) -> dict:
    """Check if the question can be answered using text + OCR content
    (matches text-only graph RAG ingestion).

    Note on packing:
      - Cap = 200000 chars (~50k tokens; safe within Qwen3.5-35B's 65k limit
        after ~1k prompt overhead + 512-token output reservation).
      - This covers ~90% of docs without truncation. Top-5% longest docs (>250k
        chars) will still be truncated; that's an inherent context-window limit.
      - all_text_contents arrives already ordered "text chunks first, OCR last"
        (see collect_text_contents_v2). This ensures full body text is included
        before OCR is appended, so v2 never has less text than v1.
    """
    CONTEXT_CAP = 200000  # was 8000 in v1, raised to cover most docs
    combined = ""
    for t in all_text_contents:
        if len(combined) + len(t) > CONTEXT_CAP:
            break
        combined += t + "\n\n"

    if not combined.strip():
        return {"answerable": False, "answer_matches": False, "predicted_answer": None, "reason": "no text"}

    prompt = TEXT_ONLY_PROMPT.format(question=question, content=combined)

    try:
        result = call_llm(client, prompt, model)
        answerable = result.get("answerable", False)
        predicted = result.get("answer")
        matches = answer_matches(str(predicted), gold_answer) if answerable and predicted else False
        return {
            "answerable": answerable,
            "answer_matches": matches,
            "predicted_answer": predicted,
            "reason": result.get("reason", ""),
        }
    except Exception as e:
        return {"answerable": False, "answer_matches": False, "predicted_answer": None, "reason": str(e)}


def validate_modality(client: OpenAI, mh: dict, all_text_contents: list[str],
                      visual_page_ids: list[int], doc_name: str, model: str) -> dict:
    """Check if a cross-modal question truly requires multiple modalities."""
    if not mh.get("cross_modal", False):
        mh["modality_filter"] = "skip_not_cross_modal"
        return mh

    # Check 1: Text+OCR-only
    text_check = check_text_only(client, mh["question"], mh["final_answer"], all_text_contents, model)
    mh["text_only_check"] = text_check  # keep field name for compatibility

    if text_check["answer_matches"]:
        mh["modality_filter"] = "fail_text_only_sufficient"
        return mh

    # Check 2: Visual-only
    visual_check = check_visual_only(client, mh["question"], mh["final_answer"],
                                     visual_page_ids, doc_name, model)
    mh["visual_only_check"] = visual_check

    if visual_check["answer_matches"]:
        mh["modality_filter"] = "fail_visual_only_sufficient"
        return mh

    mh["modality_filter"] = "pass"
    return mh


# ============================================================
# Main change: text-content collection now includes OCR
# ============================================================

def collect_text_contents_v2(layouts: list[dict]) -> list[str]:
    """Collect text content as seen by text-only graph RAG (text + OCR'd figure/table).

    Returns: text chunks first, then OCR chunks. This ordering ensures full
    body text is preserved when downstream truncates by character cap.

    This matches the content of text_chunks_original/doc_*.json — the input
    stream for LightRAG, HippoRAG, MS-GraphRAG. Caption / vlm_text are
    intentionally excluded because they are not part of text_chunks_original.
    """
    text_chunks: list[str] = []
    ocr_chunks: list[str] = []
    for l in layouts:
        modality = l.get("modality")
        if modality == "text":
            text = l.get("text", "").strip()
            if len(text) >= 30:
                text_chunks.append(text)
        elif modality in ("figure", "table"):
            # OCR'd content of figure/table layouts (what lightrag/hipporag actually see)
            ocr = l.get("ocr_text", "").strip()
            if ocr:
                # Tag with modality so the LLM judge knows where it came from
                ocr_chunks.append(f"[{modality} OCR text] {ocr}")
    # Order: text first (so it never gets pushed out by OCR), OCR after
    return text_chunks + ocr_chunks


def process_doc(client: OpenAI, doc: dict, model: str):
    doc_id = doc["doc_id"]

    input_path = os.path.join(FILTERED_SHORTCUT_DIR, f"doc{doc_id}_multihop_filtered.json")
    if not os.path.exists(input_path):
        print(f"  [SKIP] {input_path} not found")
        return

    with open(input_path) as f:
        questions = json.load(f)

    cross_modal = [q for q in questions if q.get("cross_modal", False)]
    text_only_qs = [q for q in questions if not q.get("cross_modal", False)]

    if not cross_modal:
        out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_final.json")
        with open(out_path, "w") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
        print(f"Doc {doc_id}: {len(questions)} questions, 0 cross-modal → skip modality check")
        return

    # ↓↓↓ KEY DIFFERENCE FROM v1: include OCR ↓↓↓
    all_text_contents = collect_text_contents_v2(doc["layouts"])

    visual_page_ids = sorted(set(
        l["page_id"] for l in doc["layouts"]
        if l["modality"] in ("figure", "table")
    ))

    doc_name = doc["doc_name"]

    # Stats: text vs OCR composition for transparency
    n_text_chunks = sum(1 for c in all_text_contents if not c.startswith("[figure") and not c.startswith("[table"))
    n_ocr_chunks = sum(1 for c in all_text_contents if c.startswith("[figure") or c.startswith("[table"))
    print(f"Doc {doc_id}: checking {len(cross_modal)} cross-modal questions "
          f"(text+OCR: {n_text_chunks} text + {n_ocr_chunks} OCR chunks)")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(validate_modality, client, mh, all_text_contents,
                          visual_page_ids, doc_name, model): mh["mh_id"]
            for mh in cross_modal
        }
        checked = []
        for future in as_completed(futures):
            checked.append(future.result())

    passed_cm = [q for q in checked if q["modality_filter"] == "pass"]
    fail_text = [q for q in checked if q["modality_filter"] == "fail_text_only_sufficient"]
    fail_visual = [q for q in checked if q["modality_filter"] == "fail_visual_only_sufficient"]

    print(f"  → {len(passed_cm)} passed, {len(fail_text)} failed (text+OCR-only sufficient), "
          f"{len(fail_visual)} failed (visual-only sufficient)")

    final = text_only_qs + passed_cm
    final.sort(key=lambda x: -x["quality_score"])

    out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_final.json")
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    failed_cm = fail_text + fail_visual
    if failed_cm:
        rej_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_rejected_modality.json")
        with open(rej_path, "w") as f:
            json.dump(failed_cm, f, indent=2, ensure_ascii=False)

    cm_final = sum(1 for q in final if q.get("cross_modal", False))
    print(f"  → Final: {len(final)} questions ({cm_final} cross-modal)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", type=int, default=None)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL,
                        help="LLM endpoint (default: local vLLM at 8010). "
                             "For OpenRouter use https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY,
                        help="API key (default: EMPTY for local; pass real key for OpenRouter)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="Model name (default: qwen3.5-35b for local; "
                             "qwen/qwen3.5-35b-a3b for OpenRouter)")
    args = parser.parse_args()

    with open(LOADED_INFO_PATH) as f:
        data = json.load(f)

    _patch_openrouter(args.base_url)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    print(f"=== Step 6 v2 Modality Filter (OCR-aware) ===")
    print(f"Endpoint: {args.base_url}")
    print(f"Model:    {args.model}")
    print(f"Output:   {OUTPUT_DIR}")
    print()

    if args.doc_id is not None:
        docs = [d for d in data if d["doc_id"] == args.doc_id]
    else:
        docs = data

    for doc in docs:
        process_doc(client, doc, args.model)

    print("\nDone!")


if __name__ == "__main__":
    main()
