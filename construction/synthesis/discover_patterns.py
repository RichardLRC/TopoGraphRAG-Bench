"""
Cross-Source Pattern Discovery

From the facts collected about each candidate entity, identify
higher-order patterns that require aggregating three or more facts
-- trends, contradictions, causal implications, group comparisons,
or panoramic summaries. The required fact set is recorded
explicitly so that downstream synthesis QAs can be grounded in
specific evidence.

Input:  synthesis_outputs/doc{id}_facts.json
Output: synthesis_outputs/doc{id}_patterns.json

Usage:
    python discover_patterns.py                # all docs
    python discover_patterns.py --doc-id 0     # single doc
"""

import json, os, glob, argparse
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
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "synthesis_outputs")
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "prompts")

VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://localhost:8014/v1")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "EMPTY")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3.5-35b")
NUM_WORKERS = 16

with open(os.path.join(PROMPT_DIR, "pattern_discovery.txt")) as f:
    PATTERN_PROMPT = f.read()


def call_llm(client: OpenAI, prompt: str) -> dict:
    resp = client.chat.completions.create(
        model=VLM_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=1024,
    )
    text = resp.choices[0].message.content.strip()
    if "</think>" in text: text = text.split("</think>")[-1].strip()
    if "```json" in text: text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text: text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def format_facts(facts: list[dict]) -> str:
    lines = []
    for f in facts:
        s = f["source"]
        lines.append(f'[{f["fact_id"]}] ({s["modality"]}, page {s["page_id"]}): {f["fact"]}')
    return "\n".join(lines)


def discover_patterns(client: OpenAI, entity_data: dict) -> dict | None:
    entity = entity_data["entity"]
    facts = entity_data["facts"]

    formatted = format_facts(facts)
    prompt = PATTERN_PROMPT.format(entity=entity, formatted_facts=formatted)

    try:
        result = call_llm(client, prompt)
        if not result.get("valid", False):
            return None

        required_ids = result.get("required_fact_ids", [])
        if len(required_ids) < 3:
            return None

        # Check cross-modal requirement
        required_facts = [f for f in facts if f["fact_id"] in required_ids]
        modalities = set(f["source"]["modality"] for f in required_facts)

        return {
            "entity": entity,
            "pattern": result["pattern"],
            "pattern_type": result.get("pattern_type", "unknown"),
            "required_fact_ids": required_ids,
            "required_facts": required_facts,
            "why_multi_source": result.get("why_multi_source", ""),
            "num_required": len(required_ids),
            "modalities": sorted(modalities),
        }
    except:
        return None


def process_doc(client: OpenAI, doc_id: int):
    facts_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_facts.json")
    if not os.path.exists(facts_path):
        return

    with open(facts_path) as f:
        entities_facts = json.load(f)

    if not entities_facts:
        return

    results = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(discover_patterns, client, ef): ef["entity"]
            for ef in entities_facts
        }
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    print(f"Doc {doc_id}: {len(results)} patterns from {len(entities_facts)} entities")

    out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_patterns.json")
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

    _patch_openrouter(args.base_url)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    if args.doc_id is not None:
        process_doc(client, args.doc_id)
    else:
        for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "doc*_facts.json"))):
            doc_id = int(f.split("doc")[1].split("_")[0])
            process_doc(client, doc_id)

    print("\nDone!")


if __name__ == "__main__":
    main()
