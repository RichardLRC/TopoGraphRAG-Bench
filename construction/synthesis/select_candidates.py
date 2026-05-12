"""
Candidate Entity Selection for Synthesis

Select shared entities that are rich enough to support multi-source
synthesis -- entities appearing in five or more layouts of the same
document, with at least two distinct modalities represented. These
entities become the seeds for per-layout fact extraction.

Input:  ../shared_entities/doc{id}_shared_entities.json
Output: synthesis_outputs/doc{id}_candidates.json

Usage:
    python select_candidates.py                # all docs
    python select_candidates.py --doc-id 0     # single doc
"""

import json, os, glob, argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
SHARED_ENTITIES_DIR = os.path.join(PARENT_DIR, "shared_entities")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "synthesis_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MIN_LAYOUTS = 5
MAX_CANDIDATES_PER_DOC = 20


def process_doc(doc_id: int):
    input_path = os.path.join(SHARED_ENTITIES_DIR, f"doc{doc_id}_shared_entities.json")
    if not os.path.exists(input_path):
        return

    with open(input_path) as f:
        shared = json.load(f)

    candidates = []
    for s in shared:
        if not s["is_named_entity"]:
            continue
        if s["num_layouts"] < MIN_LAYOUTS:
            continue

        candidates.append({
            "entity": s["canonical_name"],
            "aliases": s["aliases"],
            "num_layouts": s["num_layouts"],
            "modalities": s["modalities"],
            "appearances": s["appearances"],
        })

    # Sort by num_layouts descending (richer entities first for synthesis)
    candidates.sort(key=lambda x: -x["num_layouts"])
    candidates = candidates[:MAX_CANDIDATES_PER_DOC]

    print(f"Doc {doc_id}: {len(candidates)} synthesis candidates")

    out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_candidates.json")
    with open(out_path, "w") as f:
        json.dump(candidates, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", type=int, default=None)
    args = parser.parse_args()

    if args.doc_id is not None:
        process_doc(args.doc_id)
    else:
        for f in sorted(glob.glob(os.path.join(SHARED_ENTITIES_DIR, "doc*_shared_entities.json"))):
            doc_id = int(f.split("doc")[1].split("_")[0])
            process_doc(doc_id)

    print("\nDone!")


if __name__ == "__main__":
    main()
