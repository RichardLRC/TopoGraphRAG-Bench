"""
Entity Extraction from Document Layouts

For each layout with substantive content, extract named entities --
people, organizations, demographic groups, dates, metrics, and
numerical values. Visual layouts are processed alongside their
caption and OCR text so that chart-internal entities are captured.

Input:  loaded_info.json
Output: entities/doc{id}_entities.json

Usage:
    python extract_entities.py                # all docs
    python extract_entities.py --doc-id 0     # single doc
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
PROMPT_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "prompts", "entity_extraction.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "entities")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Load prompt template
# ============================================================
with open(PROMPT_PATH) as f:
    PROMPT_TEMPLATE = f.read()


# ============================================================
# Helpers
# ============================================================

def get_layout_content(layout: dict) -> str | None:
    """Get text content for a layout. Returns None if not enough content."""
    modality = layout["modality"]

    if modality == "text":
        text = layout.get("text", "").strip()
        if len(text) < 50:
            return None
        return text

    elif modality in ("figure", "table"):
        parts = []
        if layout.get("caption"):
            parts.append(f"Caption: {layout['caption']}")
        if layout.get("topic"):
            parts.append(f"Topic: {layout['topic']}")
        if layout.get("vlm_text"):
            parts.append(f"Description: {layout['vlm_text']}")
        if layout.get("ocr_text", "").strip():
            parts.append(f"Text in image: {layout['ocr_text'].strip()}")

        content = "\n".join(parts)
        if len(content) < 50:
            return None
        return content

    return None


DATA_ROOT = os.environ.get("TOPOGRAPHRAG_DATA_ROOT", "data")
PAGE_IMAGE_DIR = os.environ.get("TOPOGRAPHRAG_PAGE_IMAGE_DIR", os.path.join(DATA_ROOT, "page_images"))


def load_layout_image_b64(layout: dict, doc_name: str) -> str | None:
    """Load layout's image as base64. Use layout image if exists, else page image."""
    # Try layout-level image first. image_path in loaded_info.json is stored
    # relative to DATA_ROOT (e.g., "layout_images/foo.jpg").
    img_path = layout.get("image_path", "")
    if img_path:
        full = img_path if os.path.isabs(img_path) else os.path.join(DATA_ROOT, img_path)
        if os.path.exists(full):
            with open(full, "rb") as f:
                return base64.b64encode(f.read()).decode()
    # Fallback: page image
    base_name = doc_name.replace(".pdf", "")
    page_path = os.path.join(PAGE_IMAGE_DIR, f"{base_name}_{layout['page_id']}.jpg")
    if os.path.exists(page_path):
        with open(page_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def call_llm(client: OpenAI, prompt: str, image_b64: str = None) -> list[dict]:
    """Call LLM (with optional image) and parse JSON response."""
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
        max_tokens=2048,
    )
    text = resp.choices[0].message.content.strip()

    # Clean response
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    result = json.loads(text)
    if not isinstance(result, list):
        raise ValueError(f"Expected list, got {type(result)}")
    return result


def process_layout(client: OpenAI, layout: dict, doc_name: str = "") -> dict | None:
    """Extract entities from one layout."""
    content = get_layout_content(layout)
    if content is None:
        return None

    prompt = PROMPT_TEMPLATE.format(
        modality=layout["modality"],
        page_id=layout["page_id"],
        layout_id=layout["layout_id"],
        content=content,
    )

    # For figure/table: send the image too
    image_b64 = None
    if layout["modality"] in ("figure", "table"):
        image_b64 = load_layout_image_b64(layout, doc_name)

    for attempt in range(MAX_RETRIES):
        try:
            entities = call_llm(client, prompt, image_b64)
            # Deduplicate by name
            seen = set()
            unique = []
            for e in entities:
                name = e.get("name", "").strip()
                if name and name.lower() not in seen:
                    seen.add(name.lower())
                    unique.append({
                        "name": name,
                        "type": e.get("type", "unknown"),
                    })
            return {
                "layout_id": layout["layout_id"],
                "page_id": layout["page_id"],
                "modality": layout["modality"],
                "entities": unique,
            }
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
            else:
                print(f"  [FAIL] layout {layout['layout_id']}: {e}")
                return None


def process_doc(client: OpenAI, doc: dict) -> list[dict]:
    """Extract entities from all layouts in a document."""
    doc_id = doc["doc_id"]
    doc_name = doc["doc_name"]
    layouts = doc["layouts"]

    # Filter eligible layouts
    eligible = [l for l in layouts if get_layout_content(l) is not None]
    fig_count = sum(1 for l in eligible if l["modality"] in ("figure", "table"))
    print(f"Doc {doc_id}: {len(eligible)}/{len(layouts)} eligible layouts ({fig_count} figure/table with images)")

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_layout, client, l, doc_name): l["layout_id"]
            for l in eligible
        }
        for future in as_completed(futures):
            result = future.result()
            if result and result["entities"]:
                results.append(result)

    # Sort by layout_id
    results.sort(key=lambda x: x["layout_id"])

    total_entities = sum(len(r["entities"]) for r in results)
    print(f"  → {len(results)} layouts with entities, {total_entities} total entities")

    return results


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
    print(f"Loaded {len(data)} docs")

    _patch_openrouter(args.base_url)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    docs = [d for d in data if d["doc_id"] == args.doc_id] if args.doc_id is not None else data

    for doc in docs:
        results = process_doc(client, doc)

        out_path = os.path.join(OUTPUT_DIR, f"doc{doc['doc_id']}_entities.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
