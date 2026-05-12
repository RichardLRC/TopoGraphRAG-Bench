"""
Evidence Necessity Validation for Synthesis QAs

Two checks are applied to every synthesis question:

  1. Leave-one-out necessity: for each evidence point, check whether
     the remaining evidence still supports the synthesis conclusion.
     Points whose removal preserves the conclusion are dropped.

  2. Text-RAG simulation: retrieve top-K text chunks over the entire
     document via vector similarity and ask the LLM whether the
     conclusion can be derived from these chunks alone. Questions
     that pass this check are discarded as text-recoverable.

Input:  synthesis_outputs/doc{id}_synthesis_qa.json
Output: synthesis_outputs/doc{id}_synthesis_final.json

Usage:
    python validate_evidence.py --api-key <api-key>                       # all docs
    python validate_evidence.py --api-key <api-key> --doc-ids 0 1 3 4 5
    python validate_evidence.py --api-key <api-key> --workers 32
"""

import argparse
import asyncio
import json
import os
import glob
from typing import Optional

import numpy as np
from openai import AsyncOpenAI, OpenAI

# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
LOADED_INFO_PATH = os.path.join(PARENT_DIR, "loaded_info.json")
INPUT_DIR = os.path.join(SCRIPT_DIR, "synthesis_outputs")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "synthesis_outputs")
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_DIR)), "prompts")

# Defaults
DEFAULT_LLM_URL = "https://openrouter.ai/api/v1"
DEFAULT_LLM_MODEL = "qwen/qwen3.5-35b-a3b"
DEFAULT_EMBED_URL = "http://localhost:8001/v1"
DEFAULT_EMBED_MODEL = "qwen3-embedding"

MIN_REQUIRED_EVIDENCE = 3
RETRIEVAL_TOP_K = 20
LLM_TIMEOUT = 120

with open(os.path.join(PROMPT_DIR, "evidence_necessity.txt")) as f:
    NECESSITY_PROMPT = f.read()


# ============================================================
# OpenRouter patching
# ============================================================

def patch_openrouter(llm_url: str):
    """Inject reasoning=False + provider pinning for Qwen on OpenRouter."""
    if "openrouter" not in llm_url.lower():
        return
    from openai.resources.chat.completions import AsyncCompletions
    _orig = AsyncCompletions.create
    async def _patched(self, *args, **kwargs):
        extra = kwargs.get("extra_body") or {}
        extra.setdefault("reasoning", {"enabled": False})
        extra.setdefault("provider", {
            "order": ["Parasail", "Venice"],
            "ignore": ["Alibaba"],
            "allow_fallbacks": True,
        })
        kwargs["extra_body"] = extra
        return await _orig(self, *args, **kwargs)
    AsyncCompletions.create = _patched


# ============================================================
# Async LLM call helpers
# ============================================================

async def call_llm_async(client: AsyncOpenAI, model: str, prompt: str,
                          max_tokens: int = 256) -> Optional[dict]:
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            ),
            timeout=LLM_TIMEOUT,
        )
        text = resp.choices[0].message.content.strip()
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception:
        return None


def embed_sync(client: OpenAI, model: str, texts: list, batch: int = 16) -> np.ndarray:
    """Sync embedding (local vLLM is fast, no need for async)."""
    out = []
    for i in range(0, len(texts), batch):
        chunk = [t[:7000] for t in texts[i:i+batch]]
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend([d.embedding for d in resp.data])
    return np.array(out)


# ============================================================
# Per-QA validation
# ============================================================

async def check_evidence_necessity(client: AsyncOpenAI, model: str, qa: dict, sem: asyncio.Semaphore) -> dict:
    """Leave-one-out: for each evidence point, check if removing it still allows the conclusion."""
    question = qa["question"]
    conclusion = qa["answer"]["synthesis_conclusion"]
    evidence_points = qa["answer"]["evidence_points"]

    async def check_one(i, ep):
        async with sem:
            remaining = [e for j, e in enumerate(evidence_points) if j != i]
            remaining_str = "\n".join(
                f'- ({e["source_modality"]}, page {e["source_page"]}): {e["fact"]}'
                for e in remaining
            )
            prompt = NECESSITY_PROMPT.format(
                question=question, conclusion=conclusion,
                remaining_evidence=remaining_str,
            )
            result = await call_llm_async(client, model, prompt)
            # On LLM failure: default to "evidence required" (False = NOT sufficient
            # without it) so we don't silently drop every QA when the API is broken.
            if result is None:
                return ep, False, True  # required=True, llm_failed=True
            return ep, bool(result.get("sufficient", False)), False

    results = await asyncio.gather(*[check_one(i, ep) for i, ep in enumerate(evidence_points)])

    required, optional = [], []
    n_llm_failed = 0
    for ep, sufficient_without, llm_failed in results:
        if llm_failed:
            n_llm_failed += 1
        if sufficient_without:
            optional.append(ep)
        else:
            required.append(ep)

    qa["evidence_validation"] = {
        "required": [e["fact_id"] for e in required],
        "optional": [e["fact_id"] for e in optional],
        "num_required": len(required),
        "n_llm_failed": n_llm_failed,
    }
    qa["answer"]["evidence_points"] = required
    qa["num_sources_required"] = len(required)
    return qa


TEXT_RAG_PROMPT = """You are simulating a text-only RAG system that has ONLY text chunks (no figures, tables, or images).

## Question:
{question}

## Target conclusion the system must reach:
{conclusion}

## Retrieved text chunks (top-{k} by vector similarity over the entire document text):
{chunks}

Question: Can you derive the target conclusion using ONLY the text chunks above?
- "yes" if all key facts in the conclusion can be supported by the text
- "no" if the text is missing critical information (numbers, entities, comparisons) that's only in figures/tables

Return JSON only:
{{"sufficient": true/false, "missing": "<what's missing if no, brief>"}}"""


async def check_text_rag_can_answer(
    llm_client: AsyncOpenAI, llm_model: str,
    qa: dict, chunk_texts: list, chunk_embs: np.ndarray, query_emb: np.ndarray,
    sem: asyncio.Semaphore,
) -> dict:
    """Simulate text-RAG retrieval: top-K via cosine, then ask LLM if sufficient."""
    if not chunk_texts or chunk_embs.size == 0:
        return {"sufficient": False, "missing": "no_text_chunks"}

    chunk_embs_n = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9)
    query_emb_n = query_emb / (np.linalg.norm(query_emb) + 1e-9)
    sims = chunk_embs_n @ query_emb_n
    top_idx = np.argsort(-sims)[:RETRIEVAL_TOP_K]
    retrieved = [chunk_texts[i] for i in top_idx]
    retrieved_str = "\n\n---\n\n".join(retrieved)

    prompt = TEXT_RAG_PROMPT.format(
        question=qa["question"],
        conclusion=qa["answer"]["synthesis_conclusion"],
        k=len(retrieved),
        chunks=retrieved_str,
    )
    async with sem:
        result = await call_llm_async(llm_client, llm_model, prompt, max_tokens=512)
    if result is None:
        return {"sufficient": False, "missing": "llm_failed", "retrieved_n": len(retrieved)}
    return {
        "sufficient": bool(result.get("sufficient", False)),
        "missing": result.get("missing", ""),
        "retrieved_n": len(retrieved),
    }


async def score_synthesis(client: AsyncOpenAI, model: str, qa: dict, sem: asyncio.Semaphore) -> dict:
    with open(os.path.join(PROMPT_DIR, "synthesis_score.txt")) as f:
        score_template = f.read()
    evidence_str = "\n".join(
        f'- [{e["source_modality"]}] p{e["source_page"]}: {e["fact"]}'
        for e in qa["answer"]["evidence_points"]
    )
    prompt = score_template.format(
        question=qa["question"],
        conclusion=qa["answer"]["synthesis_conclusion"],
        evidence=evidence_str,
    )
    async with sem:
        scores = await call_llm_async(client, model, prompt)
    if scores:
        dims = ["question_clarity", "conclusion_quality", "multi_source_necessity"]
        qa["quality_scores"] = {d: scores.get(d, 0) for d in dims}
        qa["quality_score"] = min(scores.get(d, 0) for d in dims)
    else:
        qa["quality_scores"] = {}
        qa["quality_score"] = 5.0
    return qa


async def validate_qa(
    llm_client: AsyncOpenAI, llm_model: str,
    qa: dict, chunk_texts: list, chunk_embs: np.ndarray, query_emb: np.ndarray,
    sem: asyncio.Semaphore,
) -> dict:
    """v2 validation: necessity + text-RAG check (ALL QAs, not just cross_modal)."""

    qa = await check_evidence_necessity(llm_client, llm_model, qa, sem)

    if qa["evidence_validation"]["num_required"] < MIN_REQUIRED_EVIDENCE:
        qa["filter_result"] = "fail_insufficient_required_evidence"
        return qa

    required_modalities = set(e["source_modality"] for e in qa["answer"]["evidence_points"])
    qa["cross_modal"] = len(required_modalities) > 1
    qa["evidence_modalities"] = sorted(required_modalities)

    text_rag_result = await check_text_rag_can_answer(
        llm_client, llm_model, qa, chunk_texts, chunk_embs, query_emb, sem
    )
    qa["text_rag_check"] = text_rag_result

    if text_rag_result["sufficient"]:
        qa["filter_result"] = "fail_text_rag_sufficient"
        return qa

    qa = await score_synthesis(llm_client, llm_model, qa, sem)
    qa["filter_result"] = "pass"
    return qa


# ============================================================
# Per-doc orchestration
# ============================================================

def get_text_chunks_for_doc(doc_info: dict) -> list:
    """Extract text-modality content from a doc's layouts."""
    chunks = []
    for layout in doc_info.get("layouts", []):
        if layout.get("modality") == "text":
            txt = (layout.get("text") or "") + " " + (layout.get("ocr_text") or "")
            txt = txt.strip()
            if txt:
                chunks.append({
                    "layout_id": layout.get("layout_id"),
                    "page_id": layout.get("page_id"),
                    "content": txt,
                })
    return chunks


async def process_doc(
    doc_id: int, all_docs: list,
    llm_client: AsyncOpenAI, llm_model: str,
    embed_client: OpenAI, embed_model: str,
    sem: asyncio.Semaphore,
) -> Optional[dict]:
    qa_path = os.path.join(INPUT_DIR, f"doc{doc_id}_synthesis_qa.json")
    if not os.path.exists(qa_path):
        return None
    with open(qa_path) as f:
        qas = json.load(f)
    if not qas:
        return None

    doc_info = next((d for d in all_docs if d["doc_id"] == doc_id), None)
    if not doc_info:
        return None

    text_chunks = get_text_chunks_for_doc(doc_info)
    chunk_texts = [c["content"] for c in text_chunks]
    print(f"  doc_{doc_id}: {len(qas)} QAs, {len(chunk_texts)} text chunks; embedding chunks...", flush=True)

    # Embed chunks ONCE per doc
    if chunk_texts:
        chunk_embs = embed_sync(embed_client, embed_model, chunk_texts)
    else:
        chunk_embs = np.zeros((0, 4096))

    # Embed all questions in one batch
    questions = [qa["question"] for qa in qas]
    if questions:
        q_embs = embed_sync(embed_client, embed_model, questions)
    else:
        q_embs = np.zeros((0, 4096))

    print(f"  doc_{doc_id}: embeddings done; running validations...", flush=True)

    # Process QAs concurrently within this doc
    tasks = [
        validate_qa(llm_client, llm_model, qa, chunk_texts, chunk_embs, q_embs[i], sem)
        for i, qa in enumerate(qas)
    ]
    out = await asyncio.gather(*tasks)

    pass_n = sum(1 for q in out if q["filter_result"] == "pass")
    fail_text = sum(1 for q in out if q["filter_result"] == "fail_text_rag_sufficient")
    fail_ev = sum(1 for q in out if q["filter_result"] == "fail_insufficient_required_evidence")

    out_path = os.path.join(OUTPUT_DIR, f"doc{doc_id}_synthesis_final.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"  doc_{doc_id}: pass={pass_n}, fail_text_rag={fail_text}, fail_ev={fail_ev} -> {out_path}", flush=True)
    return {"doc_id": doc_id, "pass": pass_n, "fail_text_rag": fail_text, "fail_ev": fail_ev}


# ============================================================
# Main
# ============================================================

async def main_async(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    patch_openrouter(args.base_url)
    llm_client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    embed_client = OpenAI(base_url=args.embed_url, api_key="EMPTY")

    if args.doc_ids:
        doc_ids = list(args.doc_ids)
    else:
        files = glob.glob(os.path.join(INPUT_DIR, "doc*_synthesis_qa.json"))
        doc_ids = sorted(int(os.path.basename(f).split("_")[0].replace("doc", "")) for f in files)

    print(f"Processing {len(doc_ids)} docs (workers={args.workers})")
    print(f"  LLM:    {args.model} @ {args.base_url}")
    print(f"  EMBED:  {args.embed_model} @ {args.embed_url}")
    print(f"  OUTPUT: {OUTPUT_DIR}")
    print()

    with open(LOADED_INFO_PATH) as f:
        all_docs = json.load(f)

    sem = asyncio.Semaphore(args.workers)

    # Process docs sequentially (each doc concurrently within itself)
    results = []
    for d in doc_ids:
        r = await process_doc(d, all_docs, llm_client, args.model,
                               embed_client, args.embed_model, sem)
        if r:
            results.append(r)

    total_pass = sum(r["pass"] for r in results)
    total_fail_text = sum(r["fail_text_rag"] for r in results)
    total_fail_ev = sum(r["fail_ev"] for r in results)
    total = total_pass + total_fail_text + total_fail_ev
    if total == 0:
        print("\nNo QAs processed.")
        return
    print(f"\n=== SUMMARY ===")
    print(f"  Docs processed: {len(results)}")
    print(f"  Total QAs: {total}")
    print(f"  Pass:                         {total_pass:4d} ({total_pass/total*100:5.1f}%)")
    print(f"  Fail (text-RAG sufficient):   {total_fail_text:4d} ({total_fail_text/total*100:5.1f}%)")
    print(f"  Fail (insufficient evidence): {total_fail_ev:4d} ({total_fail_ev/total*100:5.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-ids", type=int, nargs="+", default=None,
                    help="Specific doc IDs (default: all in synthesis_outputs/)")
    ap.add_argument("--workers", type=int, default=24,
                    help="Concurrent LLM calls (default 24, good for OpenRouter)")
    ap.add_argument("--base-url", default=DEFAULT_LLM_URL)
    ap.add_argument("--model", default=DEFAULT_LLM_MODEL)
    ap.add_argument("--embed-url", default=DEFAULT_EMBED_URL)
    ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ap.add_argument("--api-key", required=True, help="API key (OpenRouter or EMPTY for local)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
