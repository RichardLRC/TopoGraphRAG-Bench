"""
Atomic Single-Hop QA Generation

For each shared entity, generate atomic single-hop QA pairs from
the layouts in which it appears. Two roles are produced: bridge
questions whose answer is the shared entity (described indirectly)
and target questions that mention the shared entity and ask about
one of its attributes. These atomic units are the building blocks
composed into bridge-chain questions in the next step.

Each QA is generated in three stages -- question, answer, quality
score -- and discarded if the score falls below threshold.

Input:  shared_entities/doc{id}_shared_entities.json + loaded_info.json
Output: atomic_qa/doc{id}_single_hop.json

Usage:
    python generate_atomic_qa.py                # all docs
    python generate_atomic_qa.py --doc-id 0     # single doc
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
SCORE_THRESHOLD = 8.0
MAX_ENTITIES_PER_DOC = 30       # max shared entities per doc
MAX_LAYOUTS_PER_ENTITY = 6      # max layouts per entity for QA generation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOADED_INFO_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "loaded_info.json")
SHARED_ENTITIES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "shared_entities")
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "atomic_qa")
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "prompts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load prompts
with open(os.path.join(PROMPT_DIR, "single_hop_question.txt")) as f:
    Q_PROMPT = f.read()
with open(os.path.join(PROMPT_DIR, "single_hop_answer.txt")) as f:
    A_PROMPT = f.read()
with open(os.path.join(PROMPT_DIR, "single_hop_score.txt")) as f:
    S_PROMPT = f.read()

# Role instructions for bridge vs target questions
BRIDGE_INSTRUCTION = """Generate a BRIDGE question: the answer should be "{entity}" itself.
The question must describe "{entity}" indirectly (by its semantic meaning, role, or factual attributes) WITHOUT using the name "{entity}" directly.

Good examples (use semantic/factual descriptions):
- "Which group had the highest unemployment rate in 2015?" (answer: Hispanic)
- "What institution conducted this national survey?" (answer: Pew Research Center)
- "What labor market indicator is compared for Hispanic and non-Hispanic workers?" (answer: unemployment rate)

Bad examples (do NOT use visual/layout descriptions):
- "What metric is shown on the y-axis ranging from 0% to 14%?" ← describes chart appearance, not meaning
- "Which group is represented by the brown line?" ← describes visual encoding, not the group's identity
- "What is shown in the figure on the left side?" ← references document layout"""

TARGET_INSTRUCTION_TEXT = """Generate a TARGET question: the question should mention "{entity}" by name, and ask about one of its attributes or factual details.
Examples of good target questions:
- "What was the unemployment rate for Hispanic adults in 2015?" (answer: 12%)
- "How many people did Pew Research Center survey?" (answer: 1,500)
- "According to the report, what trend has been observed among Latinos since 2008?" (answer: more optimistic)"""

TARGET_INSTRUCTION_FIGURE = """Generate a TARGET question: the question should mention "{entity}" by name, and ask about a SPECIFIC data value visible in the chart/figure/table.

You MUST pick one of these question types (choose randomly):

a) **Value lookup**: Ask for a specific value.
   "What was the personal finance satisfaction rate for Hispanic adults in 2015?"

b) **Change calculation**: Ask for the numerical change between two time points.
   "By how many percentage points did Hispanic confidence in personal finances change between 2008 and 2015?"

c) **Trend**: Ask about the overall trend or pattern.
   "How did the personal finance satisfaction rate for Hispanic adults change from 2004 to 2015?"

d) **Comparison**: Ask which of two entities has a higher/lower value.
   "Did Hispanic or General public have a higher financial satisfaction rate in 2015?"

e) **Rank**: Ask about the entity's position relative to others.
   "Among all demographic subgroups, did Hispanic adults have the highest or lowest satisfaction rate in 2015?"

Pick whichever type best fits the available data. The answer must require reading the figure/chart/table."""


# ============================================================
# Helpers
# ============================================================

def get_layout_content(layout: dict) -> str | None:
    modality = layout["modality"]
    if modality == "text":
        text = layout.get("text", "").strip()
        return text if len(text) >= 50 else None
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
        return content if len(content) >= 50 else None
    return None


PAGE_IMAGE_DIR = os.environ.get("TOPOGRAPHRAG_PAGE_IMAGE_DIR", os.path.join(os.environ.get("TOPOGRAPHRAG_DATA_ROOT", "data"), "page_images"))


def load_page_image_b64(doc_name: str, page_id: int) -> str | None:
    base_name = doc_name.replace(".pdf", "")
    path = os.path.join(PAGE_IMAGE_DIR, f"{base_name}_{page_id}.jpg")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def call_llm(client: OpenAI, prompt: str, max_tokens: int = 1024,
             image_b64: str = None) -> str:
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


def generate_single_hop(client: OpenAI, layout: dict, entity: str,
                        role: str, doc_id: int, sh_counter: list,
                        doc_text_contents: list[str] = None,
                        aliases: list[str] = None,
                        doc_name: str = "") -> dict | None:
    """Generate one single-hop QA: question → answer → score."""
    content = get_layout_content(layout)
    if content is None:
        return None

    # Check entity or any alias appears in content
    content_lower = content.lower()
    found_name = None
    names_to_check = [entity] + (aliases or [])
    for name in names_to_check:
        if name.lower() in content_lower:
            found_name = name
            break
    if not found_name:
        return None
    # Use the name variant that actually appears in the content
    entity = found_name

    modality = layout["modality"]
    page_id = layout["page_id"]
    layout_id = layout["layout_id"]

    if role == "bridge":
        role_instruction = BRIDGE_INSTRUCTION.format(entity=entity)
    elif modality in ("figure", "table"):
        role_instruction = TARGET_INSTRUCTION_FIGURE.format(entity=entity)
    else:
        role_instruction = TARGET_INSTRUCTION_TEXT.format(entity=entity)

    # Load page image for figure/table layouts
    image_b64 = None
    if modality in ("figure", "table") and doc_name:
        image_b64 = load_page_image_b64(doc_name, page_id)

    # Step 3a: Generate question (with image for figure/table)
    q_prompt = Q_PROMPT.format(
        modality=modality, page_id=page_id, layout_id=layout_id,
        content=content, entity=entity, question_role=role,
        role_instruction=role_instruction,
    )
    try:
        question = call_llm(client, q_prompt, max_tokens=256, image_b64=image_b64).strip().strip('"')
    except Exception:
        return None

    if not question or len(question) < 10:
        return None

    # Step 3b: Generate answer (with image for figure/table)
    a_prompt = A_PROMPT.format(
        modality=modality, page_id=page_id, layout_id=layout_id,
        content=content, question=question,
    )
    try:
        a_text = call_llm(client, a_prompt, max_tokens=256, image_b64=image_b64)
        a_json = parse_json(a_text)
        answer = a_json.get("answer")
        answer_type = a_json.get("answer_type", "phrase")
    except Exception:
        return None

    if not answer:
        return None

    # Step 3c: Quality score (with image for figure/table)
    s_prompt = S_PROMPT.format(
        modality=modality, content=content[:500],
        question=question, answer=answer,
    )
    try:
        s_text = call_llm(client, s_prompt, max_tokens=128, image_b64=image_b64)
        s_json = parse_json(s_text)
        score = float(s_json.get("score", 0))
    except Exception:
        score = 5.0  # Default if scoring fails

    # Figure/table targets are harder to generate, use lower threshold
    threshold = SCORE_THRESHOLD - 1.0 if (modality in ("figure", "table") and role == "target") else SCORE_THRESHOLD
    if score < threshold:
        return None

    # Step 3d: Text redundancy check (for figure/table target questions)
    text_redundant = False
    if modality in ("figure", "table") and role == "target" and doc_text_contents:
        answer_lower = answer.lower().strip()
        entity_lower = entity.lower().strip()
        for text_content in doc_text_contents:
            text_lower = text_content.lower()
            if entity_lower in text_lower and answer_lower in text_lower:
                text_redundant = True
                break

    sh_counter[0] += 1
    return {
        "sh_id": f"doc{doc_id}_sh_{sh_counter[0]:04d}",
        "doc_id": doc_id,
        "shared_entity": entity,
        "question_role": role,
        "question": question,
        "answer": answer,
        "answer_type": answer_type,
        "source": {
            "modality": modality,
            "layout_id": layout_id,
            "page_id": page_id,
        },
        "quality_score": score,
        "text_redundant": text_redundant,
    }


def process_entity(client: OpenAI, entity_info: dict, layouts_map: dict,
                   doc_id: int, sh_counter: list,
                   doc_text_contents: list[str] = None,
                   doc_name: str = "") -> list[dict]:
    """Generate single-hop QAs for one shared entity across its layouts.

    Ensures both text and figure/table modalities get questions
    for cross-modal composition.
    """
    entity_name = entity_info["canonical_name"]
    aliases = entity_info.get("aliases", [])
    appearances = entity_info["appearances"]

    # Split by modality to ensure coverage
    text_apps = [a for a in appearances if a["modality"] == "text"]
    fig_apps = [a for a in appearances if a["modality"] in ("figure", "table")]

    # Take up to MAX_LAYOUTS_PER_ENTITY, but ensure figure gets slots
    max_per_modality = MAX_LAYOUTS_PER_ENTITY
    selected_text = text_apps[:max_per_modality]
    selected_fig = fig_apps[:max_per_modality]

    results = []

    for app in selected_text + selected_fig:
        layout = layouts_map.get(app["layout_id"])
        if layout is None:
            continue

        is_fig = layout["modality"] in ("figure", "table")

        # Generate bridge question (answer = entity)
        bridge = generate_single_hop(
            client, layout, entity_name, "bridge", doc_id, sh_counter,
            doc_text_contents, aliases, doc_name
        )
        if bridge:
            results.append(bridge)

        # Generate target question (mentions entity, asks attribute)
        # For figure/table: try multiple times with temperature variation
        attempts = 3 if is_fig else 1
        for _ in range(attempts):
            target = generate_single_hop(
                client, layout, entity_name, "target", doc_id, sh_counter,
                doc_text_contents, aliases, doc_name
            )
            if target:
                results.append(target)
                break

    return results


def process_doc(client: OpenAI, doc: dict):
    """Process one document."""
    doc_id = doc["doc_id"]
    doc_name = doc["doc_name"]

    # Load shared entities
    se_path = os.path.join(SHARED_ENTITIES_DIR, f"doc{doc_id}_shared_entities.json")
    if not os.path.exists(se_path):
        print(f"  [SKIP] {se_path} not found")
        return

    with open(se_path) as f:
        shared_entities = json.load(f)

    # Filter: only named entities with 3+ layouts (2-layout entities have too little to work with)
    entities = [s for s in shared_entities if s["is_named_entity"] and s["num_layouts"] >= 3]
    # Sort: cross-modal first, then by layout count ascending (smaller = more unique questions)
    entities.sort(key=lambda x: (-x["cross_modal"], x["num_layouts"]))
    entities = entities[:MAX_ENTITIES_PER_DOC]

    print(f"Doc {doc_id}: processing {len(entities)} shared entities")

    # Build layout lookup
    layouts_map = {l["layout_id"]: l for l in doc["layouts"]}

    # Collect all text contents for redundancy check
    doc_text_contents = [
        l["text"].strip() for l in doc["layouts"]
        if l["modality"] == "text" and len(l.get("text", "").strip()) >= 30
    ]

    sh_counter = [0]
    all_results = []

    # Process entities in parallel
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_entity, client, ent, layouts_map, doc_id, sh_counter, doc_text_contents, doc_name): ent["canonical_name"]
            for ent in entities
        }
        for future in as_completed(futures):
            entity_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"  [ERR] {entity_name}: {e}")

    bridges = sum(1 for r in all_results if r["question_role"] == "bridge")
    targets = sum(1 for r in all_results if r["question_role"] == "target")
    print(f"  → {len(all_results)} QAs ({bridges} bridge, {targets} target)")

    # Sort by sh_id
    all_results.sort(key=lambda x: x["sh_id"])

    out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_single_hop.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


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
        process_doc(client, doc)

    print("\nDone!")


if __name__ == "__main__":
    main()
