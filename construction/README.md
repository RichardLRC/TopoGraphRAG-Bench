# Benchmark Construction Pipeline

This directory contains the bottom-up construction pipeline that produces
the TopoGraphRAG-Bench benchmark from a corpus of multimodal documents.
The pipeline runs in three phases mapped onto three subdirectories:

- **Shared preprocessing** (`./`): caption extraction, entity extraction,
  and cross-layout entity linking. Both downstream pipelines consume the
  shared entities produced here.
- **Multi-hop pipeline** (`./multihop/`): atomic single-hop QA generation,
  bridge-chain composition, shortcut filtering, modality necessity
  filtering, and final review.
- **Synthesis pipeline** (`./synthesis/`): candidate selection, fact
  extraction, cross-source pattern discovery, structured QA generation,
  and evidence necessity validation.

## Directory Layout

```
construction/
├── extract_captions.py            Caption / topic extraction for figures and tables
├── extract_entities.py            Per-layout named entity extraction
├── link_entities.py               Cross-layout entity merging (shared entities)
├── multihop/
│   ├── generate_atomic_qa.py      Atomic single-hop QA generation (bridge / target roles)
│   ├── compose_bridge_chain.py    Two-hop bridge-chain composition
│   ├── filter_shortcut.py         Single-source shortcut filter
│   ├── filter_modality.py         Modality necessity filter (OCR-aware)
│   └── final_check.py             Final LLM-based quality review
└── synthesis/
    ├── select_candidates.py       Candidate entity selection (>= 5 layouts)
    ├── extract_facts.py           Per-layout fact extraction
    ├── discover_patterns.py       Cross-source pattern discovery
    ├── generate_qa.py             Synthesis question + structured answer generation
    └── validate_evidence.py       Evidence necessity + text-RAG validation
```

## LLM Endpoint: Local or API

Every script that calls an LLM accepts the same three flags:

```
--base-url   LLM endpoint URL
--api-key    API key (use "EMPTY" for local vLLM without auth)
--model      Model identifier
```

Two execution modes are supported. No code change is required to switch
between them.

### Mode 1 — Local vLLM (default)

If a local vLLM server is running at `http://localhost:8010/v1`, no flags
are needed:

```bash
python extract_captions.py --doc-id 0
```

Defaults can also be overridden via environment variables:

| Flag           | Default                       | Env var fallback  |
| -------------- | ----------------------------- | ----------------- |
| `--base-url`   | `http://localhost:8010/v1`    | `VLM_BASE_URL`    |
| `--api-key`    | `EMPTY`                       | `VLM_API_KEY`     |
| `--model`      | `qwen3.5-35b`                 | `VLM_MODEL`       |

### Mode 2 — OpenRouter API

Point any script at OpenRouter by overriding the three flags:

```bash
python extract_captions.py --doc-id 0 \
    --base-url https://openrouter.ai/api/v1 \
    --api-key sk-or-v1-... \
    --model qwen/qwen3.5-35b-a3b
```

When the base URL contains `openrouter`, every request is auto-patched:

- `reasoning.enabled = false` — disables Qwen thinking mode for
  consistency across requests.
- `provider.ignore = ["Alibaba"]` — pins the request away from the
  Alibaba provider to avoid silent provider-dependent behavior shifts;
  the explicit `provider.order` falls back to Parasail and Venice.

The patching is a no-op for non-OpenRouter URLs, so local runs are
unaffected.

## Pipeline Execution Order

Run the scripts in the order below to construct the benchmark from
scratch. Each script accepts `--doc-id <N>` for single-document
processing or no flag to process the full corpus.

### Shared preprocessing

```bash
python extract_captions.py        # writes captions back into loaded_info.json
python extract_entities.py        # → entities/
python link_entities.py           # → shared_entities/
```

### Multi-hop branch

```bash
cd multihop
python generate_atomic_qa.py      # → atomic_qa/
python compose_bridge_chain.py    # → bridge_chains/
python filter_shortcut.py         # → filtered_shortcut/
python filter_modality.py         # → filtered_modality/
python final_check.py             # → final_multihop/
```

### Synthesis branch

```bash
cd synthesis
python select_candidates.py       # → synthesis_outputs/
python extract_facts.py           # → synthesis_outputs/
python discover_patterns.py       # → synthesis_outputs/
python generate_qa.py             # → synthesis_outputs/
python validate_evidence.py --api-key <key>   # → synthesis_outputs/
```

`validate_evidence.py` additionally requires a local embedding server
for the text-RAG simulation step (`--embed-url`, default
`http://localhost:8001/v1`).

## Data Dependencies

- **`loaded_info.json`** at `construction/loaded_info.json` —
  parsed document layouts with raw text, OCR text, and VLM-generated
  descriptions for each figure and table.
- **Page images directory** — JPEG renderings of each document page,
  used by scripts that pass images to a vision LLM. Path is set via the
  `TOPOGRAPHRAG_PAGE_IMAGE_DIR` env var (default: `data/page_images/`).
- **Prompts** at `../prompts/` — 17 prompt templates loaded at module
  import time. A missing template surfaces as a `FileNotFoundError` on
  import rather than mid-run.

## Outputs

All multi-hop intermediate outputs are written to
`construction/<stage>/` (e.g., `construction/atomic_qa/`).
All synthesis intermediate outputs are written to
`construction/synthesis/synthesis_outputs/`.
The final benchmark is assembled from `final_multihop/` and
`synthesis_outputs/<doc>_synthesis_final.json`.
