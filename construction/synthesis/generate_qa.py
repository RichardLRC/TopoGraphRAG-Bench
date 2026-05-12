"""
Synthesis Question and Structured Answer Generation

For each discovered pattern, generate a synthesis question and a
structured answer consisting of evidence points linked back to
specific layouts and a 1-3 sentence synthesis conclusion that cites
data drawn from these evidence points.

Input:  synthesis_outputs/doc{id}_patterns.json
Output: synthesis_outputs/doc{id}_synthesis_qa.json

Usage:
    python generate_qa.py                # all docs
    python generate_qa.py --doc-id 0     # single doc
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

with open(os.path.join(PROMPT_DIR, "synthesis_question.txt")) as f:
    Q_PROMPT = f.read()
with open(os.path.join(PROMPT_DIR, "synthesis_answer.txt")) as f:
    A_PROMPT = f.read()


def call_llm(client: OpenAI, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=VLM_MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=1024,
    )
    text = resp.choices[0].message.content.strip()
    if "</think>" in text: text = text.split("</think>")[-1].strip()
    return text


def parse_json(text: str) -> dict:
    if "```json" in text: text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text: text = text.split("```")[1].split("```")[0].strip()
    return json.loads(text)


def format_facts(facts: list[dict]) -> str:
    lines = []
    for f in facts:
        s = f["source"]
        lines.append(f'[{f["fact_id"]}] ({s["modality"]}, page {s["page_id"]}): {f["fact"]}')
    return "\n".join(lines)


def format_evidence(facts: list[dict]) -> str:
    lines = []
    for f in facts:
        s = f["source"]
        lines.append(
            f'- ({s["modality"]}, page {s["page_id"]}): {f["fact"]}'
            f'  [data: {", ".join(f.get("data_points", []))}]'
        )
    return "\n".join(lines)


def generate_synthesis_qa(client: OpenAI, pattern: dict, doc_id: int,
                          sq_id: str) -> dict | None:
    entity = pattern["entity"]
    required_facts = pattern["required_facts"]

    # Step 1: Generate question
    q_prompt = Q_PROMPT.format(
        entity=entity,
        pattern=pattern["pattern"],
        pattern_type=pattern["pattern_type"],
        formatted_required_facts=format_facts(required_facts),
    )
    try:
        question = call_llm(client, q_prompt).strip().strip('"')
    except:
        return None

    if not question or len(question) < 20:
        return None

    # Step 2: Generate structured answer
    a_prompt = A_PROMPT.format(
        question=question,
        formatted_evidence=format_evidence(required_facts),
    )
    try:
        answer_raw = call_llm(client, a_prompt)
        answer_json = parse_json(answer_raw)
        conclusion = answer_json.get("synthesis_conclusion", "")
        key_data = answer_json.get("key_data_cited", [])
    except:
        return None

    if not conclusion:
        return None

    # Build evidence_points
    evidence_points = []
    for f in required_facts:
        evidence_points.append({
            "fact_id": f["fact_id"],
            "source_layout": f["source"]["layout_id"],
            "source_page": f["source"]["page_id"],
            "source_modality": f["source"]["modality"],
            "fact": f["fact"],
            "data_points": f.get("data_points", []),
        })

    modalities = set(f["source"]["modality"] for f in required_facts)

    return {
        "sq_id": sq_id,
        "doc_id": doc_id,
        "entity": entity,
        "question": question,
        "answer": {
            "evidence_points": evidence_points,
            "synthesis_conclusion": conclusion,
            "key_data_cited": key_data,
        },
        "pattern_type": pattern["pattern_type"],
        "num_sources_required": len(required_facts),
        "cross_modal": len(modalities) > 1,
        "evidence_modalities": sorted(modalities),
    }


def process_doc(client: OpenAI, doc_id: int):
    patterns_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_patterns.json")
    if not os.path.exists(patterns_path):
        return

    with open(patterns_path) as f:
        patterns = json.load(f)

    if not patterns:
        return

    results = []
    counter = [0]

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        for pattern in patterns:
            counter[0] += 1
            sq_id = f"doc{doc_id}_syn_{counter[0]:04d}"
            futures[executor.submit(generate_synthesis_qa, client, pattern, doc_id, sq_id)] = pattern

        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    cm = sum(1 for r in results if r["cross_modal"])
    print(f"Doc {doc_id}: {len(results)} synthesis QAs ({cm} cross-modal)")

    out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_synthesis_qa.json")
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
        for f in sorted(glob.glob(os.path.join(OUTPUT_DIR, "doc*_patterns.json"))):
            doc_id = int(f.split("doc")[1].split("_")[0])
            process_doc(client, doc_id)

    print("\nDone!")


if __name__ == "__main__":
    main()
