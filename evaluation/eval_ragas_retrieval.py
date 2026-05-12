"""
Retrieval Evaluation using RAGAs
================================

Mirrors the architecture of generation/eval_ragas.py but focused on retrieval-side
metrics.

Two LLM-as-judge metrics from the RAGAs library (with enriched `reference` covering
the full reasoning chain, see build_enriched_reference below):

  1. Context Precision — fraction of retrieved context relevant to the question
  2. Context Recall    — fraction of reference info covered by retrieved context

One non-LLM benchmark-specific metric (NOT using RAGAs' ContextEntityRecall, which
suffers from degenerate LLM loops and misses benchmark-structured entities):

  3. Entity Recall — fraction of benchmark ground-truth entities (bridge_entities
                     + each reasoning_trace hop's result + synthesis data_points)
                     that appear as substrings in retrieved_context. Zero-LLM,
                     directly leverages our hop-level annotations.

All three are unit-agnostic — they treat retrieved_context as a text blob, so
they're fair across heterogeneous RAG systems (entity / triple / chunk /
community-based retrieval).

VisRAG is skipped by default because its retrieved_context contains page image
placeholders ("[Page 12 - visual content]") rather than text, so RAGAs text-based
judges cannot score it fairly. VisRAG retrieval is evaluated at page-level
(hit@k / recall@k) separately. Use --include-visrag to override.

Usage:
    python eval_ragas_retrieval.py --systems lightrag --doc-ids 0 1 3
    python eval_ragas_retrieval.py --all
    python eval_ragas_retrieval.py --fill-missing
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent       # TopoGraphRAG-Bench/evaluation/
PROJECT_ROOT = SCRIPT_DIR.parent                   # TopoGraphRAG-Bench/
# Retrieval results land alongside generation results, in a separate folder:
#   evaluation/results_retrieval/{system}/doc_<id>.json
OUTPUT_DIR = SCRIPT_DIR / "results_retrieval"

# Reuse shared utilities from eval_ragas.py: same answer-subdir auto-detect,
# same paired-data shape, same LLM defaults.
sys.path.insert(0, str(SCRIPT_DIR))
from eval_ragas import (  # noqa: E402
    load_paired, discover_doc_ids,
    DEFAULT_LLM_URL, DEFAULT_LLM_MODEL, DEFAULT_LLM_KEY,
)

DEFAULT_SYSTEMS = ["hipporag", "lightrag", "msgraphrag", "raganything_mix", "megarag"]

# ============================================================
# Constants
# ============================================================

METRIC_KEYS = [
    "context_precision",
    "context_recall",
    "entity_recall",
]

METRIC_LABELS = [
    ("Context Precision", "context_precision"),
    ("Context Recall",    "context_recall"),
    ("Entity Recall",     "entity_recall"),
]

CONTEXT_MAX_CHARS = 24000  # ~6K tokens, matches faithfulness cap in eval_ragas.py


# ============================================================
# RAGAS setup
# ============================================================

def setup_retrieval_scorers(llm_url, llm_model, llm_key):
    """Initialize RAGAs retrieval-side scorers with vLLM-compatible endpoint.

    Only ContextPrecision and ContextRecall are RAGAs metrics.
    Entity Recall is computed by compute_benchmark_entity_recall (no LLM).
    """
    from openai import AsyncOpenAI
    from openai.resources.chat.completions import AsyncCompletions
    from ragas.llms import llm_factory
    from ragas.metrics.collections import ContextPrecision, ContextRecall

    # For OpenRouter Qwen models: disable thinking + pin provider (same as eval_ragas.py)
    if "openrouter" in llm_url.lower():
        _orig_create = AsyncCompletions.create

        async def _patched_create(self, *args, **kwargs):
            extra = kwargs.get("extra_body") or {}
            extra.setdefault("reasoning", {"enabled": False})
            extra.setdefault("provider", {
                "order": ["Parasail", "Venice"],
                "ignore": ["Alibaba"],
                "allow_fallbacks": True,
            })
            kwargs["extra_body"] = extra
            return await _orig_create(self, *args, **kwargs)

        AsyncCompletions.create = _patched_create

    llm_client = AsyncOpenAI(base_url=llm_url, api_key=llm_key)
    llm = llm_factory(llm_model, provider="openai", client=llm_client, max_tokens=8192)

    return {
        "context_precision": ContextPrecision(llm=llm),
        "context_recall":    ContextRecall(llm=llm),
    }


# ============================================================
# Benchmark-specific entity recall (non-LLM)
# ============================================================

def _collect_benchmark_entities(gt_item: dict) -> List[str]:
    """Pull ground-truth entities from the benchmark's structured hop annotations.

    Sources (in priority order):
      - bridge_entities          (bridge_chain multi-hop)
      - reasoning_trace[i].result (each hop's intermediate answer)
      - answer.evidence_points[i].data_points (synthesis)
      - final_answer              (always, as backstop)
    """
    entities: List[str] = []

    for e in gt_item.get("bridge_entities") or []:
        if e:
            entities.append(str(e))

    for hop in gt_item.get("reasoning_trace") or []:
        r = hop.get("result")
        if r:
            entities.append(str(r))

    answer_obj = gt_item.get("answer") or {}
    if isinstance(answer_obj, dict):
        for ev in answer_obj.get("evidence_points") or []:
            for dp in ev.get("data_points") or []:
                if dp:
                    entities.append(str(dp))

    fa = gt_item.get("final_answer")
    if fa:
        entities.append(str(fa))

    # Dedup (case-insensitive), preserve order, drop empty
    seen = set()
    uniq = []
    for e in entities:
        e = e.strip()
        if not e:
            continue
        key = e.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq


def compute_benchmark_entity_recall(gt_item: dict, retrieved_context: str) -> Optional[float]:
    """Check how many benchmark ground-truth entities appear in retrieved_context
    (case-insensitive substring match). Non-LLM, deterministic.

    Returns None if benchmark provides no entities for this question
    (e.g., gt_item was empty); caller can treat None as "not applicable".
    """
    entities = _collect_benchmark_entities(gt_item)
    if not entities:
        return None
    ctx_lower = retrieved_context.lower()
    hits = sum(1 for e in entities if e.lower() in ctx_lower)
    return round(hits / len(entities), 4)


# ============================================================
# Reference enrichment for multi-hop coverage
# ============================================================
#
# Default RAGAs usage passes `reference = gt_answer` (a short final answer).
# For multi-hop questions, gt_answer only names the final-hop answer, so
# ContextRecall / ContextEntityRecall only test whether the final-hop evidence
# is retrieved — bridge-hop retrieval goes untested.
#
# We enrich `reference` with the benchmark's structured reasoning metadata
# (reasoning_trace, bridge_entities for bridge_chain; evidence_points for
# synthesis) so RAGAs's built-in claim decomposition and entity extraction
# cover the full reasoning chain. The RAGAs scoring logic is unchanged; only
# the input `reference` is augmented.

def build_enriched_reference(gt_answer: str, gt_item: dict) -> str:
    """Construct a reference that covers the full reasoning chain, not just
    the final answer. See module-level comment above for rationale."""
    qt = gt_item.get("question_type", "")
    lines = [f"Final answer: {gt_answer}"]

    if qt == "bridge_chain":
        trace = gt_item.get("reasoning_trace") or []
        if trace:
            lines.append("")
            lines.append("Reasoning chain:")
            for h in trace:
                sub_q = (h.get("sub_question") or "").strip()
                result = (h.get("result") or "").strip()
                hop_no = h.get("hop")
                if sub_q and result:
                    lines.append(f"  Hop {hop_no}: {sub_q} -> {result}")
        bridges = gt_item.get("bridge_entities") or []
        if bridges:
            lines.append("")
            lines.append(f"Bridge entities: {', '.join(str(b) for b in bridges)}")

    elif qt == "synthesis":
        answer_obj = gt_item.get("answer") or {}
        evs = answer_obj.get("evidence_points") or []
        if evs:
            lines.append("")
            lines.append("Required evidence points:")
            for i, ev in enumerate(evs, 1):
                fact = (ev.get("fact") or "").strip()
                if fact:
                    lines.append(f"  {i}. {fact}")

    # single_hop: gt_answer alone is sufficient (no multi-hop blind spot)

    return "\n".join(lines)


# ============================================================
# Per-question evaluation
# ============================================================

async def evaluate_question_retrieval(item: dict, scorers: dict) -> dict:
    """Score the 3 retrieval metrics for one question. Runs scorers in parallel."""
    r = item["result"]
    question = r["query"]
    gt_answer = str(r["gt_answer"])
    reference = build_enriched_reference(gt_answer, item.get("gt_item") or {})
    retrieved_context = r.get("retrieved_context") or ""

    metrics = {
        "question_id": r["question_id"],
        "doc_id": item["doc_id"],
        "question_type": item["question_type"],
        "assigned_type": item["assigned_type"],
        "modality_type": item["modality_type"],
        "hop_path": item["hop_path"],
        "num_hops": item["num_hops"],
        "query": question,
        "gt_answer": gt_answer,
        "has_retrieved_context": bool(retrieved_context),
    }

    # Empty retrieved_context = retrieval totally failed. Score 0 honestly
    # (this is a real retrieval capability signal, not a data gap).
    if not retrieved_context:
        for key in METRIC_KEYS:
            metrics[key] = 0.0
        return metrics

    ctx = retrieved_context[:CONTEXT_MAX_CHARS]

    # Non-LLM entity recall — compute synchronously, no need to gather
    metrics["entity_recall"] = compute_benchmark_entity_recall(
        item.get("gt_item") or {}, retrieved_context
    )

    async def _context_precision():
        try:
            result = await scorers["context_precision"].ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=[ctx],
            )
            return round(float(result), 4)
        except Exception as e:
            print(f"    [ContextPrecision error] {r['question_id']}: {e}")
            return None

    async def _context_recall():
        try:
            result = await scorers["context_recall"].ascore(
                user_input=question,
                reference=reference,
                retrieved_contexts=[ctx],
            )
            return round(float(result), 4)
        except Exception as e:
            print(f"    [ContextRecall error] {r['question_id']}: {e}")
            return None

    async def _with_timeout(coro, label, timeout=300):
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            print(f"    [TIMEOUT {timeout}s] {label} for {r['question_id']}")
            return None

    cp, cr = await asyncio.gather(
        _with_timeout(_context_precision(), "ContextPrecision"),
        _with_timeout(_context_recall(), "ContextRecall"),
    )
    metrics["context_precision"] = cp
    metrics["context_recall"] = cr

    return metrics


# ============================================================
# Aggregation (uses this module's METRIC_KEYS, not eval_ragas's)
# ============================================================

def aggregate(metrics_list: List[dict]) -> dict:
    summary = {"num_questions": len(metrics_list)}
    for key in METRIC_KEYS:
        values = [m[key] for m in metrics_list
                  if m.get(key) is not None and np.isfinite(m[key])]
        if values:
            summary[key] = {
                "mean": round(float(np.mean(values)), 4),
                "std":  round(float(np.std(values)), 4),
                "n":    len(values),
            }
    return summary


def aggregate_by(metrics_list: List[dict], key_fn) -> dict:
    groups = defaultdict(list)
    for m in metrics_list:
        groups[key_fn(m)].append(m)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


def _print_metric_table(agg: dict, indent: str = "  "):
    print(f"{indent}{'Metric':<26} {'Mean':>8} {'Std':>8} {'N':>5}")
    print(f"{indent}{'-' * 49}")
    for label, key in METRIC_LABELS:
        v = agg.get(key, {})
        if isinstance(v, dict) and "mean" in v:
            print(f"{indent}{label:<26} {v['mean']:>8.4f} {v['std']:>8.4f} {v['n']:>5}")
        else:
            print(f"{indent}{label:<26} {'N/A':>8}")


def print_results(all_results: Dict[str, List[dict]]):
    print(f"\n{'=' * 72}")
    print("Retrieval Evaluation Summary")
    print(f"{'=' * 72}")
    for sys_name, metrics_list in all_results.items():
        print(f"\n[{sys_name}]  (N = {len(metrics_list)})")
        _print_metric_table(aggregate(metrics_list))


# ============================================================
# Main loop
# ============================================================

async def async_main(args):
    print("Setting up RAGAS retrieval scorers...")
    scorers = setup_retrieval_scorers(args.llm_url, args.llm_model, args.llm_key)
    print(f"  LLM: {args.llm_model} @ {args.llm_url}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, List[dict]] = {}

    for sys_name in args.systems:
        if sys_name == "visrag" and not args.include_visrag:
            print(f"\nSkipping visrag: retrieved_context is image placeholders, not text.")
            print(f"  VisRAG retrieval is evaluated at page-level separately.")
            print(f"  Use --include-visrag to force-run RAGAs context metrics on it.")
            continue

        if args.doc_ids:
            doc_ids = args.doc_ids
        else:
            doc_ids = discover_doc_ids(sys_name)

        if not doc_ids:
            print(f"  No data found for {sys_name}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Evaluating retrieval: {sys_name}  ({len(doc_ids)} docs)")

        sys_dir = OUTPUT_DIR / sys_name
        sys_dir.mkdir(parents=True, exist_ok=True)

        all_metrics: List[dict] = []

        for doc_id in doc_ids:
            doc_out = sys_dir / f"doc{doc_id}.json"

            # --fill-missing: load cache, only compute None metrics
            if args.fill_missing and doc_out.exists():
                with open(doc_out) as f:
                    cached = json.load(f)

                missing_keys = [
                    k for k in METRIC_KEYS
                    if any(m.get(k) is None for m in cached)
                ]
                if not missing_keys:
                    all_metrics.extend(cached)
                    print(f"  doc{doc_id}: all metrics complete ({len(cached)} questions)")
                    continue

                print(f"  doc{doc_id}: filling missing "
                      f"[{', '.join(missing_keys)}] for {len(cached)} questions")
                paired = load_paired(sys_name, [doc_id])
                if not paired:
                    all_metrics.extend(cached)
                    continue
                paired_by_qid = {p["result"]["question_id"]: p for p in paired}

                semaphore = asyncio.Semaphore(args.max_workers)
                done_count = 0

                async def _fill_one(idx, cached_m):
                    nonlocal done_count
                    needs_fill = {k for k in missing_keys if cached_m.get(k) is None}
                    if not needs_fill:
                        done_count += 1
                        return
                    item = paired_by_qid.get(cached_m["question_id"])
                    if not item:
                        done_count += 1
                        return
                    r = item["result"]
                    question = r["query"]
                    gt_answer = str(r["gt_answer"])
                    reference = build_enriched_reference(
                        gt_answer, item.get("gt_item") or {}
                    )
                    retrieved_context = r.get("retrieved_context") or ""

                    async with semaphore:
                        if not retrieved_context:
                            for k in needs_fill:
                                cached[idx][k] = 0.0
                            done_count += 1
                            if done_count % 5 == 0 or done_count == len(cached):
                                print(f"    Progress: {done_count}/{len(cached)}")
                            return

                        ctx = retrieved_context[:CONTEXT_MAX_CHARS]

                        async def _safe_score(key, label, coro):
                            try:
                                result = await asyncio.wait_for(coro, timeout=300)
                                cached[idx][key] = round(float(result), 4)
                            except asyncio.TimeoutError:
                                print(f"    [{label} TIMEOUT 300s] {cached_m['question_id']}")
                            except Exception as e:
                                print(f"    [{label} error] {cached_m['question_id']}: {e}")

                        fills = []
                        if "context_precision" in needs_fill:
                            fills.append(_safe_score(
                                "context_precision", "ContextPrecision",
                                scorers["context_precision"].ascore(
                                    user_input=question, reference=reference,
                                    retrieved_contexts=[ctx],
                                ),
                            ))
                        if "context_recall" in needs_fill:
                            fills.append(_safe_score(
                                "context_recall", "ContextRecall",
                                scorers["context_recall"].ascore(
                                    user_input=question, reference=reference,
                                    retrieved_contexts=[ctx],
                                ),
                            ))
                        if "entity_recall" in needs_fill:
                            # Non-LLM: synchronous, no coroutine
                            cached[idx]["entity_recall"] = compute_benchmark_entity_recall(
                                item.get("gt_item") or {}, retrieved_context
                            )
                        await asyncio.gather(*fills)
                        done_count += 1
                        if done_count % 5 == 0 or done_count == len(cached):
                            print(f"    Progress: {done_count}/{len(cached)}")

                await asyncio.gather(*[_fill_one(i, m) for i, m in enumerate(cached)])

                with open(doc_out, "w") as f:
                    json.dump(cached, f, indent=2, ensure_ascii=False, default=str)
                all_metrics.extend(cached)
                continue

            # Normal path: use cache unless --force
            if doc_out.exists() and not args.force:
                with open(doc_out) as f:
                    cached = json.load(f)
                all_metrics.extend(cached)
                print(f"  doc{doc_id}: loaded {len(cached)} from cache")
                continue

            paired = load_paired(sys_name, [doc_id])
            if not paired:
                continue
            print(f"  doc{doc_id}: {len(paired)} questions")

            semaphore = asyncio.Semaphore(args.max_workers)
            doc_metrics = [None] * len(paired)
            done_count = 0

            async def _eval_one(idx, item):
                nonlocal done_count
                async with semaphore:
                    m = await evaluate_question_retrieval(item, scorers)
                    doc_metrics[idx] = m
                    done_count += 1
                    if done_count % 5 == 0 or done_count == len(paired):
                        print(f"    Progress: {done_count}/{len(paired)}")

            await asyncio.gather(*[_eval_one(i, item) for i, item in enumerate(paired)])

            with open(doc_out, "w") as f:
                json.dump(doc_metrics, f, indent=2, ensure_ascii=False, default=str)

            all_metrics.extend(doc_metrics)

        if not all_metrics:
            continue

        all_results[sys_name] = all_metrics

        out_path = OUTPUT_DIR / f"{sys_name}_summary.json"
        output_data = {
            "system": sys_name,
            "doc_ids": doc_ids,
            "overall": aggregate(all_metrics),
            "by_question_type": aggregate_by(all_metrics, lambda m: m.get("question_type", "unknown")),
            "by_modality_type": aggregate_by(all_metrics, lambda m: m.get("modality_type", "unknown")),
            "by_hop_path":      aggregate_by(all_metrics, lambda m: m.get("hop_path", "unknown")),
            "per_question": all_metrics,
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Summary saved -> {out_path}")

    print_results(all_results)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Evaluation using RAGAs")
    parser.add_argument("--systems", nargs="+", default=DEFAULT_SYSTEMS,
                        help=f"Systems to evaluate (default: {DEFAULT_SYSTEMS})")
    parser.add_argument("--include-visrag", action="store_true",
                        help="Include visrag (RAGAs won't score it fairly because "
                             "its retrieved_context is image placeholders)")
    parser.add_argument("--doc-ids", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true", help="Run on all available docs")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--fill-missing", action="store_true",
                        help="Load cached results, only compute metrics with None values")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if cached")
    # LLM config (defaults imported from eval_ragas to stay in sync)
    parser.add_argument("--llm-url",   default=DEFAULT_LLM_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-key",   default=DEFAULT_LLM_KEY)
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
