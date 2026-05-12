"""
Per-Layout Fact Extraction

For each candidate entity, extract one concrete data-grounded fact
from every layout in which it appears. Visual layouts are processed
with the original image so that numerical values are read directly
from the chart or table rather than paraphrased. Near-duplicate
facts are removed; entities retaining fewer than three distinct
facts are dropped.

Input:  synthesis_outputs/doc{id}_candidates.json + ../loaded_info.json
Output: synthesis_outputs/doc{id}_facts.json

Usage:
    python extract_facts.py                # all docs
    python extract_facts.py --doc-id 0     # single doc
"""

import json, os, base64, glob, argparse, time
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
LOADED_INFO_PATH = os.path.join(PARENT_DIR, "loaded_info.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "synthesis_outputs")
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "prompts")
PAGE_IMAGE_DIR = os.environ.get("TOPOGRAPHRAG_PAGE_IMAGE_DIR", os.path.join(os.environ.get("TOPOGRAPHRAG_DATA_ROOT", "data"), "page_images"))

VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://localhost:8014/v1")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "EMPTY")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3.5-35b")
NUM_WORKERS = 16
MIN_UNIQUE_FACTS = 3
MAX_LAYOUTS_PER_ENTITY = 10

with open(os.path.join(PROMPT_DIR, "fact_extraction.txt")) as f:
    FACT_PROMPT = f.read()


def get_layout_content(layout: dict) -> str | None:
    modality = layout["modality"]
    if modality == "text":
        text = layout.get("text", "").strip()
        return text if len(text) >= 50 else None
    else:
        parts = []
        if layout.get("caption"): parts.append(f"Caption: {layout['caption']}")
        if layout.get("topic"): parts.append(f"Topic: {layout['topic']}")
        if layout.get("vlm_text"): parts.append(f"Description: {layout['vlm_text']}")
        if layout.get("ocr_text", "").strip(): parts.append(f"Text in image: {layout['ocr_text'].strip()}")
        content = "\n".join(parts)
        return content if len(content) >= 50 else None


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
        model=VLM_MODEL, messages=messages, temperature=0.0, max_tokens=512,
    )
    text = resp.choices[0].message.content.strip()
    if "</think>" in text: text = text.split("</think>")[-1].strip()
    if "```json" in text: text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text: text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def extract_fact(client: OpenAI, entity: str, layout: dict,
                 doc_name: str, aliases: list[str]) -> dict | None:
    content = get_layout_content(layout)
    if not content:
        return None

    # Check entity or alias in content
    content_lower = content.lower()
    found = any(a.lower() in content_lower for a in [entity] + aliases)
    if not found:
        return None

    image_b64 = None
    if layout["modality"] in ("figure", "table"):
        image_b64 = load_page_image_b64(doc_name, layout["page_id"])

    prompt = FACT_PROMPT.format(
        entity=entity, modality=layout["modality"], page_id=layout["page_id"],
        content=content,
    )

    try:
        result = call_llm(client, prompt, image_b64)
        fact = result.get("fact")
        if not fact:
            return None
        return {
            "fact": fact,
            "data_points": result.get("data_points", []),
            "source": {
                "layout_id": layout["layout_id"],
                "page_id": layout["page_id"],
                "modality": layout["modality"],
            }
        }
    except:
        return None


def deduplicate_facts(facts: list[dict]) -> list[dict]:
    """Remove facts with identical data_points."""
    seen = set()
    unique = []
    for f in facts:
        key = frozenset(str(d).lower() for d in f["data_points"])
        if key and key not in seen:
            seen.add(key)
            unique.append(f)
        elif not key:
            unique.append(f)
    return unique


def process_entity(client: OpenAI, candidate: dict, layouts_map: dict,
                   doc_name: str) -> dict | None:
    entity = candidate["entity"]
    aliases = candidate.get("aliases", [])
    appearances = candidate["appearances"][:MAX_LAYOUTS_PER_ENTITY]

    facts = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        for app in appearances:
            layout = layouts_map.get(app["layout_id"])
            if not layout:
                continue
            futures[executor.submit(extract_fact, client, entity, layout, doc_name, aliases)] = app

        for future in as_completed(futures):
            result = future.result()
            if result:
                facts.append(result)

    facts = deduplicate_facts(facts)

    if len(facts) < MIN_UNIQUE_FACTS:
        return None

    # Assign fact IDs
    for i, f in enumerate(facts):
        f["fact_id"] = f"f{i+1}"

    return {
        "entity": entity,
        "aliases": aliases,
        "facts": facts,
        "num_facts": len(facts),
    }


def process_doc(client: OpenAI, doc: dict):
    doc_id = doc["doc_id"]
    doc_name = doc["doc_name"]

    cand_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_candidates.json")
    if not os.path.exists(cand_path):
        return

    with open(cand_path) as f:
        candidates = json.load(f)

    if not candidates:
        return

    layouts_map = {l["layout_id"]: l for l in doc["layouts"]}

    results = []
    for cand in candidates:
        entity_facts = process_entity(client, cand, layouts_map, doc_name)
        if entity_facts:
            results.append(entity_facts)

    total_facts = sum(r["num_facts"] for r in results)
    print(f"Doc {doc_id}: {len(results)} entities with {total_facts} facts")

    out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_facts.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    global VLM_BASE_URL, VLM_API_KEY, VLM_MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", type=int, default=None)
    parser.add_argument("--base-url", default=VLM_BASE_URL,
                        help="LLM endpoint. Default: local vLLM. For OpenRouter: https://openrouter.ai/api/v1")
    parser.add_argument("--api-key", default=VLM_API_KEY,
                        help="EMPTY for local vLLM; real key for OpenRouter")
    parser.add_argument("--model", default=VLM_MODEL,
                        help="Model name. For OpenRouter: qwen/qwen3.5-35b-a3b")
    args = parser.parse_args()
    VLM_BASE_URL, VLM_API_KEY, VLM_MODEL = args.base_url, args.api_key, args.model

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
