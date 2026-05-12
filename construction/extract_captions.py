"""
Caption Extraction for Figures and Tables

Uses a VLM to read each page image and produce a caption and topic
for every figure and table layout. Results are written back to
loaded_info.json in place.

Usage:
    python extract_captions.py                # all docs
    python extract_captions.py --doc-id 0     # single doc
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
MAX_RETRIES = 3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOADED_INFO_PATH = os.path.join(SCRIPT_DIR, "loaded_info.json")
PAGE_IMAGE_DIR = os.environ.get("TOPOGRAPHRAG_PAGE_IMAGE_DIR", os.path.join(os.environ.get("TOPOGRAPHRAG_DATA_ROOT", "data"), "page_images"))

PROMPT_TEMPLATE = """This document page contains {num_figures} figure(s)/table(s) with these layout IDs: {layout_ids}.

For each one, I have provided a brief description to help you locate it:
{figure_hints}

For EACH figure/table, extract:
1. caption: the title/caption text near the figure (usually above or below it). Copy it exactly. If none, set null.
2. topic: one sentence describing what this figure/table measures or shows.

Return JSON array only:
[{{"layout_id": <id>, "caption": "<text or null>", "topic": "<description>"}}]"""

# ============================================================
# Core logic
# ============================================================

def load_page_image_b64(doc_name: str, page_id: int) -> str | None:
    base_name = doc_name.replace(".pdf", "")
    path = os.path.join(PAGE_IMAGE_DIR, f"{base_name}_{page_id}.jpg")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def call_vlm(client: OpenAI, image_b64: str, prompt: str) -> list[dict]:
    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        temperature=0.0,
        max_tokens=4096,
    )
    text = resp.choices[0].message.content.strip()
    # Strip think tags / markdown
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def process_page(client: OpenAI, doc_name: str, page_id: int, figures: list[dict]) -> list[dict]:
    image_b64 = load_page_image_b64(doc_name, page_id)
    if not image_b64:
        print(f"  [WARN] No page image: {doc_name} page {page_id}")
        return [{"layout_id": f["layout_id"], "caption": None, "topic": ""} for f in figures]

    layout_ids = ", ".join(str(f["layout_id"]) for f in figures)
    hints = "\n".join(
        f"  - layout_id={f['layout_id']} ({f['modality']}): {f.get('vlm_text', '')[:80]}..."
        for f in figures
    )
    prompt = PROMPT_TEMPLATE.format(
        num_figures=len(figures), layout_ids=layout_ids, figure_hints=hints
    )

    for attempt in range(MAX_RETRIES):
        try:
            results = call_vlm(client, image_b64, prompt)
            result_map = {r["layout_id"]: r for r in results}
            return [
                {"layout_id": f["layout_id"],
                 "caption": result_map.get(f["layout_id"], {}).get("caption"),
                 "topic": result_map.get(f["layout_id"], {}).get("topic", "")}
                for f in figures
            ]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [RETRY] page {page_id}: {e}")
                time.sleep(2)
            else:
                print(f"  [FAIL] page {page_id}: {e}")
                return [{"layout_id": f["layout_id"], "caption": None, "topic": ""} for f in figures]


def process_doc(client: OpenAI, doc: dict, num_workers: int = NUM_WORKERS):
    doc_id, doc_name = doc["doc_id"], doc["doc_name"]
    layouts = doc["layouts"]

    # Group figures/tables by page
    page_figures = {}
    for l in layouts:
        if l["modality"] in ("figure", "table"):
            page_figures.setdefault(l["page_id"], []).append(l)

    total = sum(len(v) for v in page_figures.values())
    if total == 0:
        return
    print(f"Doc {doc_id} ({doc_name}): {total} figures/tables, {len(page_figures)} pages")

    # Process all pages in parallel
    all_captions = {}
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_page, client, doc_name, pid, figs): pid
            for pid, figs in page_figures.items()
        }
        for future in as_completed(futures):
            for r in future.result():
                all_captions[r["layout_id"]] = {"caption": r["caption"], "topic": r["topic"]}

    # Write back to layout
    found = 0
    for l in layouts:
        if l["layout_id"] in all_captions:
            l["caption"] = all_captions[l["layout_id"]]["caption"]
            l["topic"] = all_captions[l["layout_id"]]["topic"]
            if l["caption"]:
                found += 1

    print(f"  → {found}/{total} got captions")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", type=int, default=None)
    parser.add_argument("--start-doc", type=int, default=None,
                        help="Start from this doc_id (skip earlier docs)")
    parser.add_argument("--end-doc", type=int, default=None,
                        help="End at this doc_id (exclusive)")
    parser.add_argument("--base-url", default=VLM_BASE_URL)
    parser.add_argument("--api-key", default=VLM_API_KEY,
                        help="EMPTY for local vLLM; real key for OpenRouter")
    parser.add_argument("--model", default=VLM_MODEL)
    parser.add_argument("--output", default=None,
                        help="Output path (default: overwrite loaded_info.json)")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()

    with open(LOADED_INFO_PATH) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} docs")

    _patch_openrouter(args.base_url)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.doc_id is not None:
        docs = [d for d in data if d["doc_id"] == args.doc_id]
    elif args.start_doc is not None or args.end_doc is not None:
        start = args.start_doc or 0
        end = args.end_doc or 999999
        docs = [d for d in data if start <= d["doc_id"] < end]
    else:
        docs = data
    print(f"Processing {len(docs)} docs")
    for doc in docs:
        process_doc(client, doc, num_workers=args.num_workers)

    output_path = args.output or LOADED_INFO_PATH
    print(f"\nSaving to {output_path}")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("Done!")


if __name__ == "__main__":
    main()
