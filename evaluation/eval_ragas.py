"""
Generation Evaluation using RAGAS + Step Coverage
===================================================

Metrics (RAGAS library):
  1. Answer Accuracy   — response vs GT answer (claim decomposition + NLI)
  2. Faithfulness      — response claims grounded in retrieved context
  3. Response Relevancy — response addresses the question (requires embeddings)

Custom metric:
  4. Step Coverage     — reasoning steps covered by response
                         Bridge-chain: N hops (1/N each)
                         Synthesis: N evidence points (1/N each)

Usage:
    python eval_ragas.py --systems raganything --doc-ids 0 1 3
    python eval_ragas.py --systems raganything --all
    python eval_ragas.py --all --skip-step-coverage
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
# Expected layout:
#   - benchmark file:          PROJECT_ROOT/annotations/benchmark.json
#                              (downloaded from the HuggingFace data package)
#   - per-system answer files: PROJECT_ROOT/<system>/<auto>/doc_{id}.json
#                              lightrag → mix/, hipporag/raganything_mix/megarag → default/,
#                              msgraphrag → local/, visrag → root
#   - evaluation output goes to SCRIPT_DIR/results/
# Both BENCHMARK_PATH and ANSWER_DIR can be overridden via env vars or --benchmark.
BENCHMARK_PATH = Path(os.environ.get(
    "TOPOGRAPHRAG_BENCHMARK",
    str(PROJECT_ROOT / "annotations" / "benchmark.json"),
))
ANSWER_DIR = Path(os.environ.get("TOPOGRAPHRAG_ANSWER_DIR", str(PROJECT_ROOT)))
OUTPUT_DIR = SCRIPT_DIR / "results"


def _set_paths(benchmark_path: Optional[str] = None,
               answer_dir: Optional[str] = None,
               output_dir: Optional[str] = None) -> None:
    """Override the benchmark/answer/output path globals at runtime."""
    global BENCHMARK_PATH, ANSWER_DIR, OUTPUT_DIR, _benchmark_cache
    if benchmark_path:
        BENCHMARK_PATH = Path(benchmark_path)
        _benchmark_cache = None
    if answer_dir:
        ANSWER_DIR = Path(answer_dir)
    if output_dir:
        OUTPUT_DIR = Path(output_dir)

ALL_SYSTEMS = ["raganything", "raganything_mix", "lightrag", "hipporag", "msgraphrag", "raptor", "visrag", "megarag"]

# Default server config
DEFAULT_LLM_URL = "http://localhost:8010/v1"
DEFAULT_LLM_MODEL = "qwen3.5-35b"
DEFAULT_LLM_KEY = "EMPTY"
DEFAULT_EMBED_URL = "http://localhost:8001/v1"
DEFAULT_EMBED_MODEL = "qwen3-embedding"
DEFAULT_EMBED_KEY = "EMPTY"

METRIC_KEYS = [
    "answer_accuracy",
    "faithfulness",
    "response_relevancy",
    "step_coverage",
]


# ============================================================
# RAGAS setup
# ============================================================

def setup_ragas(llm_url, llm_model, llm_key, embed_url, embed_model, embed_key,
                skip_response_relevancy=False):
    """Initialize RAGAS scorers with vLLM endpoints."""
    from openai import AsyncOpenAI
    from openai.resources.chat.completions import AsyncCompletions
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerAccuracy, Faithfulness, AnswerRelevancy

    # For OpenRouter Qwen models: disable thinking + pin provider for consistency
    if "openrouter" in llm_url.lower():
        _orig_create = AsyncCompletions.create
        async def _patched_create(self, *args, **kwargs):
            extra = kwargs.get("extra_body") or {}
            extra.setdefault("reasoning", {"enabled": False})
            extra.setdefault("provider", {"order": ["Parasail", "Venice"], "ignore": ["Alibaba"], "allow_fallbacks": True})
            kwargs["extra_body"] = extra
            return await _orig_create(self, *args, **kwargs)
        AsyncCompletions.create = _patched_create

    llm_client = AsyncOpenAI(base_url=llm_url, api_key=llm_key)
    llm = llm_factory(llm_model, provider="openai", client=llm_client, max_tokens=8192)

    scorers = {
        "answer_accuracy": AnswerAccuracy(llm=llm),
        "faithfulness": Faithfulness(llm=llm),
    }

    if not skip_response_relevancy:
        from ragas.embeddings import OpenAIEmbeddings
        embed_client = AsyncOpenAI(base_url=embed_url, api_key=embed_key)
        embeddings = OpenAIEmbeddings(client=embed_client, model=embed_model)
        scorers["response_relevancy"] = AnswerRelevancy(llm=llm, embeddings=embeddings)

    return scorers


# ============================================================
# Step Coverage
# ============================================================

def _call_llm_sync(prompt: str, api_base: str, model: str, api_key: str):
    """Synchronous LLM call for step coverage."""
    import httpx
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    if "openrouter" in api_base.lower():
        body["reasoning"] = {"enabled": False}
        body["provider"] = {"order": ["Parasail", "Venice"], "ignore": ["Alibaba"], "allow_fallbacks": True}
    for attempt in range(3):
        try:
            resp = httpx.post(
                f"{api_base.rstrip('/')}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120.0,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            m = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
            if m:
                content = m.group(1)
            for pattern in (r'\{[\s\S]*\}', r'\[[\s\S]*\]'):
                m = re.search(pattern, content)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except json.JSONDecodeError:
                        pass
            return json.loads(content)
        except Exception as e:
            if attempt == 2:
                print(f"    [Step Coverage LLM error] {e}")
    return None


STEP_PROMPT_BRIDGE = """Given a multi-hop question, its ground-truth answer, and a system response, determine which reasoning steps the response covers.

## Question: {question}
## GT Answer: {gt_answer}

## Reasoning steps:
{steps_text}

## System Response: {response}

For each step, determine if the response demonstrates knowledge of that step's result (even if not stated explicitly).

Return JSON:
{{{step_keys}}}"""


STEP_PROMPT_SYNTHESIS = """Given a synthesis question, its ground-truth conclusion, and a system response, determine which evidence points the response covers.

## Question: {question}
## GT Conclusion: {gt_answer}

## Required evidence points:
{evidence_text}

## System Response: {response}

For each evidence point, determine if the response demonstrates knowledge of or references that specific data/fact.

Return JSON:
{{{evidence_keys}}}"""


def compute_step_coverage(question: str, gt_answer: str, response: str,
                          question_type: str, gt_item: dict,
                          llm_url: str, llm_model: str, llm_key: str) -> Optional[float]:
    """Compute step coverage score."""
    if not response or not response.strip():
        return 0.0

    if question_type == "bridge_chain":
        trace = gt_item.get("reasoning_trace", [])
        if not trace:
            return None
        n = len(trace)
        steps_lines = []
        step_key_parts = []
        for i, hop in enumerate(trace):
            steps_lines.append(
                f"  Step {i+1} [{hop.get('source_modality', '')}]: "
                f"{hop.get('sub_question', '')} → {hop.get('result', '')}"
            )
            step_key_parts.append(f'"step{i+1}_covered": true/false')

        prompt = STEP_PROMPT_BRIDGE.format(
            question=question, gt_answer=gt_answer,
            steps_text="\n".join(steps_lines),
            response=response,
            step_keys=", ".join(step_key_parts),
        )
        result = _call_llm_sync(prompt, llm_url, llm_model, llm_key)
        if not result or not isinstance(result, dict):
            return None
        covered = [result.get(f"step{i+1}_covered", False) for i in range(n)]
        return round(sum(1.0 / n for c in covered if c), 4) if n > 0 else 0.0

    elif question_type == "synthesis":
        evidence_points = gt_item.get("answer", {}).get("evidence_points", [])
        if not evidence_points:
            return None
        n = len(evidence_points)
        ev_lines = []
        ev_key_parts = []
        for i, ep in enumerate(evidence_points):
            ev_lines.append(
                f"  Evidence {i+1} [{ep.get('source_modality', '')}]: {ep.get('fact', '')}"
            )
            ev_key_parts.append(f'"evidence{i+1}_covered": true/false')

        prompt = STEP_PROMPT_SYNTHESIS.format(
            question=question, gt_answer=gt_answer,
            evidence_text="\n".join(ev_lines),
            response=response,
            evidence_keys=", ".join(ev_key_parts),
        )
        result = _call_llm_sync(prompt, llm_url, llm_model, llm_key)
        if not result or not isinstance(result, dict):
            return None
        covered = [result.get(f"evidence{i+1}_covered", False) for i in range(n)]
        return round(sum(1.0 / n for c in covered if c), 4) if n > 0 else 0.0

    return None


# ============================================================
# Data loading
# ============================================================

_benchmark_cache = None


def load_benchmark() -> List[dict]:
    global _benchmark_cache
    if _benchmark_cache is None:
        with open(BENCHMARK_PATH) as f:
            _benchmark_cache = json.load(f)
    return _benchmark_cache


def load_gt_by_doc(doc_id: int) -> Dict[str, dict]:
    """Load GT items keyed by question_id for a doc."""
    benchmark = load_benchmark()
    result = {}
    for q in benchmark:
        if q["doc_id"] != doc_id:
            continue
        qid = q.get("mh_id") or q.get("sq_id") or q.get("sh_id")
        if qid:
            result[qid] = q
    return result


# Baseline systems use varying answer-subdir conventions:
#   lightrag → mix/, hipporag/megarag/raganything_mix → default/,
#   msgraphrag → local/, visrag → root.
# We auto-detect: pick the first subdir (or system root) that holds doc_<id>.json.
_ANSWER_SUBDIR_CACHE: Dict[str, Path] = {}


def _resolve_answer_dir(system: str) -> Optional[Path]:
    if system in _ANSWER_SUBDIR_CACHE:
        return _ANSWER_SUBDIR_CACHE[system]
    sys_root = ANSWER_DIR / system
    if not sys_root.exists():
        return None
    candidates = [sys_root] + [p for p in sorted(sys_root.iterdir()) if p.is_dir()]
    for cand in candidates:
        if any(re.match(r"doc_\d+\.json$", f.name) for f in cand.iterdir()):
            _ANSWER_SUBDIR_CACHE[system] = cand
            return cand
    return None


def load_results(system: str, doc_id: int) -> Optional[dict]:
    sys_dir = _resolve_answer_dir(system)
    if sys_dir is None:
        return None
    path = sys_dir / f"doc_{doc_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def discover_doc_ids(system: str) -> List[int]:
    sys_dir = _resolve_answer_dir(system)
    if sys_dir is None:
        return []
    doc_ids = set()
    for f in sys_dir.iterdir():
        # answer files: doc_<id>.json   (skip cost_doc_<id>.json)
        m = re.match(r"doc_(\d+)\.json$", f.name)
        if m:
            doc_ids.add(int(m.group(1)))
    return sorted(doc_ids)


def _derive_modality(em_list) -> str:
    em = set(em_list or [])
    em = {"text" if m in ("text", "text_only") else m for m in em}
    if not em: return "unknown"
    if em == {"text"}: return "text_only"
    if em == {"figure"}: return "figure"
    if em == {"table"}: return "table"
    if len(em) > 1: return "multimodal"
    return next(iter(em))


def load_paired(system: str, doc_ids: List[int]) -> List[dict]:
    paired = []
    for doc_id in doc_ids:
        gt_items = load_gt_by_doc(doc_id)
        sys_data = load_results(system, doc_id)
        if not sys_data:
            continue
        for r in sys_data["results"]:
            qid = r["question_id"]
            gt_item = gt_items.get(qid, {})

            # Some answer files omit gt_answer; inject from the benchmark's
            # final_answer (multihop) or answer.synthesis_conclusion (synthesis fallback)
            if "gt_answer" not in r or not r.get("gt_answer"):
                if "final_answer" in gt_item:
                    r["gt_answer"] = gt_item["final_answer"] or ""
                else:
                    ans = gt_item.get("answer")
                    if isinstance(ans, dict):
                        r["gt_answer"] = ans.get("synthesis_conclusion") or ""
                    else:
                        r["gt_answer"] = ans or ""

            # Always derive from gt_item (runner-written modality_type is unreliable;
            # e.g. megarag mislabels figure→figure as text_only).
            em = gt_item.get("evidence_modalities") or [
                s.get("modality") for s in gt_item.get("evidence_sources", [])
            ]
            modality_type = _derive_modality(em)
            if not modality_type or modality_type == "unknown":
                modality_type = r.get("modality_type", "unknown")

            qtype = r.get("question_type") or gt_item.get("question_type", "unknown")
            num_hops = r.get("num_hops", gt_item.get("num_hops", gt_item.get("num_sources_required", 2)))

            # Derive assigned_type for multihop bridge_chain
            assigned_type = r.get("assigned_type") or gt_item.get("assigned_type")
            if not assigned_type:
                if qtype == "synthesis":
                    assigned_type = "synthesis"
                elif qtype == "bridge_chain":
                    assigned_type = f"bridge_{num_hops}hop"
                else:
                    assigned_type = qtype

            paired.append({
                "doc_id": doc_id,
                "question_id": qid,
                "question_type": qtype,
                "result": r,
                "gt_item": gt_item,
                "num_hops": num_hops,
                "assigned_type": assigned_type,
                "modality_type": modality_type,
                "hop_path": gt_item.get("hop_path", r.get("hop_path", "")),
            })
    return paired


# ============================================================
# Per-question evaluation
# ============================================================

async def evaluate_question(item: dict, scorers: dict,
                            llm_url: str, llm_model: str, llm_key: str,
                            run_step_coverage: bool) -> dict:
    r = item["result"]
    question = r["query"]
    gt_answer = str(r["gt_answer"])
    response = r.get("response") or ""
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
        "response": response,
    }

    async def _answer_accuracy():
        try:
            result = await scorers["answer_accuracy"].ascore(
                user_input=question, response=response, reference=gt_answer,
            )
            return round(float(result), 4)
        except Exception as e:
            print(f"    [AnswerAccuracy error] {r['question_id']}: {e}")
            return None

    async def _faithfulness():
        if not retrieved_context:
            return None
        try:
            ctx = retrieved_context[:24000]  # ~6K tokens
            resp = response[:2500]           # ~600 tokens, caps ~15 claims
            result = await scorers["faithfulness"].ascore(
                user_input=question, response=resp,
                retrieved_contexts=[ctx],
            )
            return round(float(result), 4)
        except Exception as e:
            print(f"    [Faithfulness error] {r['question_id']}: {e}")
            return None

    async def _response_relevancy():
        if "response_relevancy" not in scorers:
            return None
        try:
            result = await scorers["response_relevancy"].ascore(
                user_input=question, response=response,
            )
            return round(float(result), 4)
        except Exception as e:
            print(f"    [ResponseRelevancy error] {r['question_id']}: {e}")
            return None

    async def _step_coverage():
        if not run_step_coverage:
            return None
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, compute_step_coverage,
            question, gt_answer, response,
            item["question_type"], item["gt_item"],
            llm_url, llm_model, llm_key,
        )

    async def _with_timeout(coro, label, timeout=600):
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            print(f"    [TIMEOUT {timeout}s] {label} for {r['question_id']}")
            return None

    aa, faith, rr, sc = await asyncio.gather(
        _with_timeout(_answer_accuracy(), "AnswerAccuracy"),
        _with_timeout(_faithfulness(), "Faithfulness"),
        _with_timeout(_response_relevancy(), "ResponseRelevancy"),
        _with_timeout(_step_coverage(), "StepCoverage"),
    )
    metrics["answer_accuracy"] = aa
    metrics["faithfulness"] = faith
    metrics["response_relevancy"] = rr
    metrics["step_coverage"] = sc

    # Generate reason for scores
    async def _generate_reason():
        import httpx
        prompt = f"""Given the following evaluation results, briefly explain why each metric got its score (1-2 sentences each). Be concise.

Question: {question}
Ground Truth Answer: {gt_answer}
System Response: {response[:500]}

Scores:
- Answer Accuracy: {aa}
- Faithfulness: {faith}
- Step Coverage: {sc}

For each metric, explain the score in 1-2 sentences."""
        body = {
            "model": llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 512,
        }
        if "openrouter" in llm_url.lower():
            body["reasoning"] = {"enabled": False}
            body["provider"] = {"order": ["Parasail", "Venice"], "ignore": ["Alibaba"], "allow_fallbacks": True}
        try:
            resp = httpx.post(
                f"{llm_url.rstrip('/')}/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {llm_key}"},
                timeout=60.0,
            )
            resp.raise_for_status()
            content = (resp.json()["choices"][0]["message"].get("content") or "").strip()
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            return content
        except Exception as e:
            return f"Error generating reason: {e}"

    reason = await _with_timeout(_generate_reason(), "Reason", timeout=60)
    metrics["reason"] = reason

    return metrics


# ============================================================
# Aggregation & Printing
# ============================================================

def aggregate(metrics_list: List[dict]) -> dict:
    summary = {"num_questions": len(metrics_list)}
    for key in METRIC_KEYS:
        values = [m[key] for m in metrics_list if m.get(key) is not None]
        if values:
            summary[key] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values)), 4),
                "n": len(values),
            }
    return summary


def aggregate_by(metrics_list: List[dict], key_fn) -> dict:
    groups = defaultdict(list)
    for m in metrics_list:
        groups[key_fn(m)].append(m)
    return {k: aggregate(v) for k, v in sorted(groups.items())}


METRIC_LABELS = [
    ("Answer Accuracy",     "answer_accuracy"),
    ("Faithfulness",        "faithfulness"),
    ("Response Relevancy",  "response_relevancy"),
    ("Step Coverage",       "step_coverage"),
]


def _print_metric_table(agg: dict, indent: str = "  "):
    print(f"{indent}{'Metric':<24} {'Mean':>8} {'Std':>8} {'N':>5}")
    print(f"{indent}{'-'*47}")
    for label, key in METRIC_LABELS:
        v = agg.get(key, {})
        if isinstance(v, dict) and "mean" in v:
            print(f"{indent}{label:<24} {v['mean']:>8.4f} {v['std']:>8.4f} {v['n']:>5}")
        else:
            print(f"{indent}{label:<24} {'N/A':>8}")


def print_results(all_results: Dict[str, List[dict]]):
    print(f"\n{'='*70}")
    print("GENERATION EVALUATION RESULTS")
    print(f"{'='*70}")

    for sys_name, mlist in all_results.items():
        print(f"\nSystem: {sys_name}  (Total N={len(mlist)})")

        print(f"\n  [Overall]")
        _print_metric_table(aggregate(mlist))

        by_qtype = aggregate_by(mlist, lambda m: m.get("question_type", "unknown"))
        for qt in ["bridge_chain", "synthesis"]:
            if qt in by_qtype:
                agg_g = by_qtype[qt]
                print(f"\n  [{qt}]  (N={agg_g['num_questions']})")
                _print_metric_table(agg_g)

        by_mod = aggregate_by(mlist, lambda m: m.get("modality_type", "unknown"))
        for mod in ["multimodal", "text_only"]:
            if mod in by_mod:
                agg_g = by_mod[mod]
                print(f"\n  [{mod}]  (N={agg_g['num_questions']})")
                _print_metric_table(agg_g)

        by_hop = aggregate_by(mlist, lambda m: m.get("hop_path", "unknown"))
        print(f"\n  [By hop_path]")
        header = f"  {'Path':<28} {'N':>4}"
        for lbl, _ in METRIC_LABELS:
            header += f" {lbl[:10]:>10}"
        print(header)
        print(f"  {'-'*(len(header)-2)}")
        for group, agg_g in by_hop.items():
            row = f"  {str(group):<28} {agg_g['num_questions']:>4}"
            for _, key in METRIC_LABELS:
                v = agg_g.get(key, {}).get("mean")
                row += f" {v:>10.4f}" if v is not None else f" {'N/A':>10}"
            print(row)


# ============================================================
# Main
# ============================================================

async def async_main(args):
    print("Setting up RAGAS scorers...")
    scorers = setup_ragas(
        args.llm_url, args.llm_model, args.llm_key,
        args.embed_url, args.embed_model, args.embed_key,
        skip_response_relevancy=args.skip_response_relevancy,
    )
    print(f"  LLM: {args.llm_model} @ {args.llm_url}")
    if not args.skip_response_relevancy:
        print(f"  Embedding: {args.embed_model} @ {args.embed_url}")
    else:
        print(f"  Skipping Response Relevancy (no embedding needed)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for sys_name in args.systems:
        if args.doc_ids:
            doc_ids = args.doc_ids
        else:
            doc_ids = discover_doc_ids(sys_name)

        if not doc_ids:
            print(f"  No data found for {sys_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {sys_name}  ({len(doc_ids)} docs)")

        sys_dir = OUTPUT_DIR / sys_name
        sys_dir.mkdir(parents=True, exist_ok=True)

        all_metrics: List[dict] = []

        for doc_id in doc_ids:
            doc_out = sys_dir / f"doc{doc_id}.json"

            # --fill-missing: load cache, only compute None metrics
            if args.fill_missing and doc_out.exists():
                with open(doc_out) as f:
                    cached = json.load(f)
                # Find which metrics are missing
                missing_keys = []
                for key in METRIC_KEYS:
                    if any(m.get(key) is None for m in cached):
                        missing_keys.append(key)
                if not missing_keys:
                    all_metrics.extend(cached)
                    print(f"  doc{doc_id}: all metrics complete ({len(cached)} questions)")
                    continue

                print(f"  doc{doc_id}: filling missing [{', '.join(missing_keys)}] for {len(cached)} questions")
                paired = load_paired(sys_name, [doc_id])
                if not paired:
                    all_metrics.extend(cached)
                    continue
                # Build lookup: question_id -> paired item
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
                    response = r.get("response") or ""
                    async with semaphore:
                        if "response_relevancy" in needs_fill and "response_relevancy" in scorers:
                            try:
                                result = await scorers["response_relevancy"].ascore(
                                    user_input=question, response=response,
                                )
                                cached[idx]["response_relevancy"] = round(float(result), 4)
                            except Exception as e:
                                print(f"    [ResponseRelevancy error] {cached_m['question_id']}: {e}")
                        if "step_coverage" in needs_fill:
                            loop = asyncio.get_event_loop()
                            sc = await loop.run_in_executor(
                                None, compute_step_coverage,
                                question, str(r["gt_answer"]), response,
                                item["question_type"], item["gt_item"],
                                args.llm_url, args.llm_model, args.llm_key,
                            )
                            cached[idx]["step_coverage"] = sc
                        if "answer_accuracy" in needs_fill:
                            try:
                                result = await scorers["answer_accuracy"].ascore(
                                    user_input=question, response=response, reference=str(r["gt_answer"]),
                                )
                                cached[idx]["answer_accuracy"] = round(float(result), 4)
                            except Exception as e:
                                print(f"    [AnswerAccuracy error] {cached_m['question_id']}: {e}")
                        if "faithfulness" in needs_fill:
                            retrieved_context = r.get("retrieved_context") or ""
                            if retrieved_context:
                                try:
                                    ctx = retrieved_context[:24000]  # ~6K tokens
                                    resp = response[:2500]           # ~600 tokens, caps ~15 claims
                                    result = await asyncio.wait_for(
                                        scorers["faithfulness"].ascore(
                                            user_input=question, response=resp,
                                            retrieved_contexts=[ctx],
                                        ),
                                        timeout=300,
                                    )
                                    cached[idx]["faithfulness"] = round(float(result), 4)
                                except asyncio.TimeoutError:
                                    print(f"    [Faithfulness TIMEOUT 300s] {cached_m['question_id']}")
                                except Exception as e:
                                    print(f"    [Faithfulness error] {cached_m['question_id']}: {e}")
                        done_count += 1
                        if done_count % 5 == 0 or done_count == len(cached):
                            print(f"    Progress: {done_count}/{len(cached)}")

                await asyncio.gather(*[_fill_one(i, m) for i, m in enumerate(cached)])

                with open(doc_out, "w") as f:
                    json.dump(cached, f, indent=2, ensure_ascii=False, default=str)
                all_metrics.extend(cached)
                continue

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
                    m = await evaluate_question(
                        item, scorers,
                        args.llm_url, args.llm_model, args.llm_key,
                        run_step_coverage=not args.skip_step_coverage,
                    )
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
            "by_hop_path": aggregate_by(all_metrics, lambda m: m.get("hop_path", "unknown")),
            "per_question": all_metrics,
        }
        with open(out_path, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Summary saved -> {out_path}")

    print_results(all_results)


def main():
    parser = argparse.ArgumentParser(description="Generation Evaluation (RAGAS + Step Coverage)")
    parser.add_argument("--systems", nargs="+", default=["raganything"])
    parser.add_argument("--doc-ids", type=int, nargs="+", default=None)
    parser.add_argument("--all", action="store_true", help="Run on all available docs")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--skip-step-coverage", action="store_true")
    parser.add_argument("--skip-response-relevancy", action="store_true",
                        help="Skip Response Relevancy (no embedding server needed)")
    parser.add_argument("--fill-missing", action="store_true",
                        help="Load cached results and only compute missing metrics")
    parser.add_argument("--force", action="store_true", help="Re-evaluate even if cached")
    # LLM config
    parser.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-key", default=DEFAULT_LLM_KEY)
    # Embedding config
    parser.add_argument("--embed-url", default=DEFAULT_EMBED_URL)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--embed-key", default=DEFAULT_EMBED_KEY)
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
