"""
Cross-Layout Entity Linking

Identify shared entities that appear in multiple layouts of the same
document, merging surface variants (Hispanic / Hispanics / Latino /
Latinos) via plural normalization, content-word overlap, and an LLM
matcher. The resulting shared entities serve as bridge anchors for
downstream question composition.

Input:  entities/doc{id}_entities.json
Output: shared_entities/doc{id}_shared_entities.json

Usage:
    python link_entities.py                # all docs
    python link_entities.py --doc-id 0     # single doc
"""

import json, os, argparse, re, time
from collections import defaultdict
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


VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://localhost:8010/v1")
VLM_API_KEY = os.environ.get("VLM_API_KEY", "EMPTY")
VLM_MODEL = os.environ.get("VLM_MODEL", "qwen3.5-35b")
NUM_WORKERS = 16

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENTITIES_DIR = os.path.join(SCRIPT_DIR, "entities")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "shared_entities")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Normalization & Alias Matching
# ============================================================

# Words to ignore when computing overlap
STOP_WORDS = {"the", "a", "an", "of", "in", "for", "and", "or", "to", "with",
              "at", "by", "on", "is", "are", "was", "were", "than", "that",
              "this", "its", "their", "from", "who", "which", "least", "most",
              "more", "less", "experience", "level"}


def normalize(name: str) -> str:
    """Basic normalization: lowercase, strip, remove trailing punctuation."""
    return re.sub(r'[.,;:!?\'"]+$', '', name.strip().lower())


def get_content_words(name: str) -> set[str]:
    """Extract meaningful content words from an entity name."""
    words = set(re.findall(r'[a-z]+', normalize(name)))
    return words - STOP_WORDS


def get_singular_plural_variants(name: str) -> set[str]:
    """Generate common singular/plural variants."""
    n = normalize(name)
    variants = {n}
    if n.endswith("s"):
        variants.add(n[:-1])
    else:
        variants.add(n + "s")
    if n.endswith("os"):
        variants.add(n[:-1])
    elif n.endswith("o"):
        variants.add(n + "s")
    return variants


def names_are_similar(name_a: str, name_b: str) -> bool:
    """Only match trivial cases: plural variants are handled by get_singular_plural_variants.
    Everything else goes to LLM matching in Phase 2."""
    return False


def llm_match(client: OpenAI, entity_a: str, entity_b: str, prompt_template: str) -> bool:
    """Use LLM to check if two entity names refer to the same thing."""
    prompt = prompt_template.format(entity_a=entity_a, entity_b=entity_b)
    try:
        resp = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=64,
        )
        text = resp.choices[0].message.content.strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        result = json.loads(text)
        return result.get("match", False)
    except Exception:
        return False


def build_alias_groups(all_entities: list[dict], client: OpenAI = None) -> dict[str, str]:
    """
    Build mapping: normalized_name → canonical_name.
    Groups variants using: 1) plural matching, 2) word overlap, 3) LLM matching.
    """
    name_counts = defaultdict(int)
    for ent in all_entities:
        name_counts[normalize(ent["name"])] += 1

    canonical = {}
    all_names = sorted(name_counts.keys())

    # Load LLM prompt
    prompt_path = os.path.join(os.path.dirname(SCRIPT_DIR), "prompts", "entity_matching.txt")
    llm_prompt_template = ""
    if client and os.path.exists(prompt_path):
        with open(prompt_path) as f:
            llm_prompt_template = f.read()

    # Phase 1: plural + word overlap (fast, no LLM)
    for name in all_names:
        if name in canonical:
            continue

        variants = get_singular_plural_variants(name)
        existing_canonical = None
        for v in variants:
            if v in canonical:
                existing_canonical = canonical[v]
                break

        if existing_canonical:
            canonical[name] = existing_canonical
            continue

        for existing_name, canon in list(canonical.items()):
            if existing_name == canon and names_are_similar(name, existing_name):
                canonical[name] = canon
                existing_canonical = canon
                break

        if existing_canonical:
            continue

        best = name
        best_count = name_counts.get(name, 0)
        for v in variants:
            if v in name_counts and name_counts[v] > best_count:
                best = v
                best_count = name_counts[v]
        canonical[name] = best
        for v in variants:
            if v in name_counts:
                canonical[v] = best

    # Phase 2: LLM matching for unmatched pairs that share at least 1 content word
    if client and llm_prompt_template:
        # Get canonical group leaders
        leaders = set(v for v in canonical.values())
        leaders = sorted(leaders)

        # Find candidate pairs: different groups but share content words
        candidates = []
        for i, a in enumerate(leaders):
            a_words = get_content_words(a)
            if len(a_words) < 1:
                continue
            for b in leaders[i+1:]:
                b_words = get_content_words(b)
                if len(b_words) < 1:
                    continue
                # Must share at least one content word AND be similar length
                if a_words & b_words:
                    len_ratio = len(a) / len(b) if len(b) > 0 else 0
                    if 0.3 < len_ratio < 3.0:  # Don't match "U.S." with "estimated % among U.S. Latino population"
                        candidates.append((a, b))

        if candidates:
            print(f"    LLM matching: {len(candidates)} candidate pairs")

            # Parallel LLM calls
            merges = []
            with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
                futures = {
                    executor.submit(llm_match, client, a, b, llm_prompt_template): (a, b)
                    for a, b in candidates
                }
                for future in as_completed(futures):
                    a, b = futures[future]
                    if future.result():
                        merges.append((a, b))

            # Apply merges: map b's group into a's group
            for a, b in merges:
                canon_a = canonical.get(a, a)
                canon_b = canonical.get(b, b)
                if canon_a == canon_b:
                    continue
                # Pick the more frequent as the new canonical
                if name_counts.get(canon_b, 0) > name_counts.get(canon_a, 0):
                    canon_a, canon_b = canon_b, canon_a
                # Remap everything pointing to canon_b → canon_a
                for k, v in list(canonical.items()):
                    if v == canon_b:
                        canonical[k] = canon_a

            if merges:
                print(f"    LLM merged: {len(merges)} pairs")

    return canonical


# ============================================================
# Core Logic
# ============================================================

def process_doc(doc_id: int, client: OpenAI = None) -> list[dict]:
    """Find shared entities in a document."""
    input_path = os.path.join(ENTITIES_DIR, f"doc{doc_id}_entities.json")
    if not os.path.exists(input_path):
        print(f"  [SKIP] {input_path} not found")
        return []

    with open(input_path) as f:
        layouts_entities = json.load(f)

    # Flatten all entities
    all_entities = []
    for le in layouts_entities:
        for ent in le["entities"]:
            all_entities.append({
                **ent,
                "layout_id": le["layout_id"],
                "page_id": le["page_id"],
                "modality": le["modality"],
            })

    if not all_entities:
        return []

    # Build alias groups (with optional LLM matching)
    canonical_map = build_alias_groups(all_entities, client)

    # Group by canonical name
    groups = defaultdict(lambda: {"aliases": set(), "appearances": [], "types": set()})

    for ent in all_entities:
        norm = normalize(ent["name"])
        canon = canonical_map.get(norm, norm)

        g = groups[canon]
        g["aliases"].add(ent["name"])
        g["types"].add(ent["type"])
        # Deduplicate same layout
        if not any(a["layout_id"] == ent["layout_id"] for a in g["appearances"]):
            g["appearances"].append({
                "layout_id": ent["layout_id"],
                "page_id": ent["page_id"],
                "modality": ent["modality"],
            })

    # Filter: must appear in 2+ different layouts
    shared = []
    for canon, g in groups.items():
        if len(g["appearances"]) < 2:
            continue

        modalities = set(a["modality"] for a in g["appearances"])
        pages = set(a["page_id"] for a in g["appearances"])
        entity_types = g["types"] - {"number", "date"}  # Prefer non-number entities

        shared.append({
            "canonical_name": canon,
            "aliases": sorted(g["aliases"]),
            "entity_types": sorted(g["types"]),
            "appearances": sorted(g["appearances"], key=lambda x: x["layout_id"]),
            "num_layouts": len(g["appearances"]),
            "modalities": sorted(modalities),
            "cross_modal": len(modalities) > 1,
            "cross_page": len(pages) > 1,
            "is_named_entity": bool(entity_types),  # Has non-number/date types
        })

    # Sort: cross-modal first, then by num_layouts descending
    shared.sort(key=lambda x: (-x["cross_modal"], -x["num_layouts"]))

    return shared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", type=int, default=None)
    parser.add_argument("--base-url", default=VLM_BASE_URL)
    parser.add_argument("--api-key", default=VLM_API_KEY,
                        help="EMPTY for local vLLM; real key for OpenRouter")
    parser.add_argument("--model", default=VLM_MODEL)
    args = parser.parse_args()

    with open(os.path.join(SCRIPT_DIR, "loaded_info.json")) as f:
        data = json.load(f)

    _patch_openrouter(args.base_url)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    doc_ids = [args.doc_id] if args.doc_id is not None else [d["doc_id"] for d in data]

    for doc_id in doc_ids:
        shared = process_doc(doc_id, client)

        named = [s for s in shared if s["is_named_entity"]]
        cross_modal = [s for s in shared if s["cross_modal"]]

        print(f"Doc {doc_id}: {len(shared)} shared entities, "
              f"{len(named)} named, {len(cross_modal)} cross-modal")

        out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_shared_entities.json")
        with open(out_path, "w") as f:
            json.dump(shared, f, indent=2, ensure_ascii=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
