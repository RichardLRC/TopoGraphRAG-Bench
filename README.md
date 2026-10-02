<div align="center">

# TopoGraphRAG-Bench

**A topology-aware benchmark for multimodal document RAG**

*Single-hop · Bridge-chain · Multi-source synthesis*

[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-TopoGraphRAG--Bench-yellow)](https://huggingface.co/datasets/diandianone123/topographrag-bench)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-red.svg)](#citation)

</div>

---

TopoGraphRAG-Bench evaluates retrieval-augmented generation (RAG) systems
on **layout-grounded multimodal documents** — text, figures, and tables
each anchored to a specific page region. Questions are organized into
three reasoning topologies and constructed bottom-up, so each instance
has provable construction-time guarantees of **shortcut resistance**,
**modality necessity**, and **evidence necessity**.

## At a Glance

| Statistic                         | Value                                    |
| --------------------------------- | ---------------------------------------- |
| **Total questions**               | 2,024                                    |
| **Documents**                     | 201 (9 domains, mean 45 pages, 13K words)|
| **Reasoning topologies**          | 3 (Single-hop / Bridge-chain / Synthesis)|
| **Single-hop / Bridge-chain / Synthesis** | 302 / 1,014 / 708                |
| **Cross-modal questions**         | 38.2%                                    |
| **Figure-or-table-required**      | 66.8%                                    |
| **Bridge-chain hop paths**        | 9                                        |

## Highlights

- **Three reasoning topologies in one benchmark.** Single-hop, bridge-chain
  multi-hop, and multi-source synthesis side by side — three structurally
  distinct evaluation regimes drawn from the same document corpus.
- **Verified at construction time, not asserted post-hoc.** Every instance
  passes filters for single-source shortcut resistance, modality necessity
  (OCR-aware), and evidence necessity (synthesis leave-one-out + text-RAG
  simulation).
- **Two-thirds need visual evidence.** 66.8% of questions cannot be answered
  by text-only retrieval — a useful stress test for multimodal vs.
  text-centric RAG pipelines.
- **Bottom-up compositional construction.** Atomic single-hop QAs are
  generated first, then composed into bridge-chain and synthesis
  questions — so the reasoning structure is guaranteed by construction.
- **Self-contained, reproducible pipeline.** Full code to regenerate or
  extend the benchmark, with a single `--base-url` switch between local
  vLLM and OpenRouter.

## Repository Structure

```
TopoGraphRAG-Bench/
├── construction/        Bottom-up benchmark construction pipeline
│   ├── multihop/        Bridge-chain QA generation (5 stages)
│   └── synthesis/       Multi-source synthesis QA generation (5 stages)
├── evaluation/          RAG evaluation: generation + retrieval, OpenRouter-aware
├── prompts/             17 LLM prompt templates used by the construction pipeline
└── README.md            (this file)
```

See [`construction/README.md`](construction/README.md) for full pipeline details.

## Quick Start

### 1. Download data

```bash
pip install huggingface_hub
hf download diandianone123/topographrag-bench --repo-type dataset --local-dir data/
cd data && for f in archives/*.tar.zst; do
    tar --use-compress-program=unzstd -xf "$f"
done
```

After extraction, your `data/` directory contains:

```
data/
├── annotations/benchmark.json    2,024 questions
├── loaded_info.json              parsed corpus consumed by construction pipeline
├── layout_images/                cropped figure/table images
├── layout_text_images/           cropped text region images
├── page_images/                  full page renderings
└── doc_pdfs/                     original PDF files
```

### 2. Load the benchmark

```python
import json

with open("data/annotations/benchmark.json") as f:
    benchmark = json.load(f)

print(f"Total questions: {len(benchmark)}")
for q in benchmark[:3]:
    print(f"[{q['question_type']}] {q['question']}")
```

Each instance carries a stable `question_id`, the natural-language
`question`, the ground-truth answer (`final_answer` or
`answer.synthesis_conclusion`), `evidence_sources` linking back to
specific `(doc_id, page_id, layout_id)` triples, and reasoning metadata
(`hop_path`, `cross_modal`, `bridge_entities`, …).

### 3. Evaluate a RAG system

Once a system has produced per-doc answer files at
`<project_root>/<system>/<auto>/doc_<id>.json`, score them with:

```bash
cd evaluation
python eval_ragas.py            --systems lightrag hipporag --all
python eval_ragas_retrieval.py  --systems lightrag hipporag --all
```

| Metric family | Metrics |
| -------- | --- |
| **Generation** | Answer Accuracy · Faithfulness · Response Relevancy · Step Coverage |
| **Retrieval**  | Context Precision · Context Recall · Entity Recall |

### 4. (Optional) Regenerate the benchmark

Every LLM-calling script supports both **local vLLM** (default) and
**OpenRouter** (override `--base-url`, `--api-key`, `--model`).
OpenRouter requests are auto-patched to disable Qwen thinking and pin
provider away from Alibaba. See
[`construction/README.md`](construction/README.md).

## Construction Methodology

The benchmark is built bottom-up. Atomic single-hop QAs are generated
around shared entities — entities that appear in multiple layouts of the
same document. Bridge-chain questions are composed from pairs of atomic
QAs sharing an entity; synthesis questions are derived from cross-source
patterns over an entity's facts. Three construction-time filters ensure
each instance genuinely requires the reasoning it claims:

| Filter | Test | Discards |
| --- | --- | --- |
| **Shortcut resistance** | Can either hop's evidence alone produce the answer? | Yes → discard |
| **Modality necessity** (OCR-aware) | Can text + OCR'd figure/table content alone answer it? | Yes → discard cross-modal label |
| **Evidence necessity** (synthesis) | Does leave-one-out break the conclusion? Can a text-RAG simulation recover it? | No / Yes → discard |

These properties are operationalized as filters, not asserted as
post-hoc labels.

## Data Card

- **Source documents**: 201 documents sampled from the MMDocIR corpus,
  spanning 9 domains (academic papers, laws, research reports, guidebooks,
  brochures, tutorials, government, financial, industry).
- **Languages**: English.
- **Question composition**: 302 single-hop · 1,014 bridge-chain (2-hop)
  · 708 synthesis.
- **Modality coverage**: 38.2% cross-modal; 66.8% require figure or table
  evidence at all.
- **Evidence per question**: 1 (single-hop) · 2 (bridge-chain) · 3.2 mean (synthesis).
- **Data hosting**: All artifacts on [HuggingFace](https://huggingface.co/datasets/diandianone123/topographrag-bench).



*(BibTeX will be finalized upon publication.)*

## License

| Asset | License |
| --- | --- |
| **Code** (this repository) | Apache 2.0 |
| **Benchmark annotations** (`annotations/benchmark.json`) | See dataset page |
| **Source documents & layout artifacts** (`archives/*`) | Upstream MMDocIR terms |

## Acknowledgements

Source documents are derived from
[MMDocIR]([https://huggingface.co/datasets/MMDocIR/MMDocIR_Eval_Dataset](https://huggingface.co/datasets/MMDocIR/MMDocRAG)).
Layout parsing relies on MinerU + LayoutLMv3. The construction pipeline
adapts ideas from MuSiQue (bottom-up compositional multi-hop), MIMG
(multi-agent QA generation with score-based filtering), and DocHop-QA
(multi-document multi-hop scientific QA).
