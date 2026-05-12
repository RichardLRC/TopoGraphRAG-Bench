#!/bin/bash
# Run six RAG baselines (lightrag, hipporag, msgraphrag, raganything, visrag,
# megarag) on a TopoGraphRAG-Bench split. The doc-id list, questions
# directory, and answer-output directory are taken from the parent shell;
# this script focuses on orchestration and uniform OpenRouter wiring.
#
# LLM (via OpenRouter):
#   text-only RAG  (lightrag/hipporag/msgraphrag): qwen/qwen3-30b-a3b-instruct-2507
#   multimodal RAG (raganything/visrag/megarag):   qwen/qwen3-vl-30b-a3b-instruct
#   Each runner auto-patches OpenRouter request bodies:
#       reasoning: { enabled: False }    # disable thinking
#       provider:  { ignore: ["Alibaba"], ... }
#
# Embedding:
#   text-only RAG (lightrag/hipporag/msgraphrag): localhost:8001 (qwen3-embedding)     [GPU 0]
#   raganything:                                   localhost:9001 (qwen3-vl-embedding)  [GPU 1]
#   visrag:    internal VisRAG-Ret encoder         (no external URL)
#   megarag:   internal Alibaba-NLP/gme-Qwen2-VL-2B-Instruct (no external URL)
#
# Indexes: each runner is invoked with --skip-indexing --use-original,
# assuming per-doc indexes have been prebuilt for the split being evaluated.
#
# Usage:
#   bash run_baselines.sh                          # all 6 systems
#   bash run_baselines.sh systems lightrag megarag # subset

set -e

EXT="${TOPOGRAPHRAG_EXPERIMENT_ROOT:-$(pwd)}"
RUN="${TOPOGRAPHRAG_BASELINE_ROOT:?Please set TOPOGRAPHRAG_BASELINE_ROOT to the baseline runner directory}"
LOG=$EXT/logs
mkdir -p $LOG

BENCH_PY="${TOPOGRAPHRAG_PYTHON:-python}"

# === OpenRouter config ===
OPENROUTER_URL="https://openrouter.ai/api/v1"
OPENROUTER_KEY="${OPENROUTER_KEY:?Please set OPENROUTER_KEY}"
TEXT_MODEL="qwen/qwen3-30b-a3b-instruct-2507"
VL_MODEL="qwen/qwen3-vl-30b-a3b-instruct"

# Text-only RAG reads LLM_* (via llm_config.py)
export LLM_BASE_URL="$OPENROUTER_URL"
export LLM_MODEL="$TEXT_MODEL"
export LLM_API_KEY="$OPENROUTER_KEY"

DOCS=$(cat $EXT/doc_ids.txt)

ALL_SYSTEMS="lightrag hipporag msgraphrag raganything visrag megarag"
SYSTEMS="$ALL_SYSTEMS"
if [ "$1" = "systems" ]; then
    shift
    SYSTEMS="$@"
fi

START=$(date +%s)

run_one() {
    local sys=$1
    local s_start=$(date +%s)
    echo
    echo "============================================================"
    echo "[$(date +%H:%M:%S)] STARTING $sys"
    echo "============================================================"

    case "$sys" in
        lightrag)
            cd $RUN/lightrag
            ./venv/bin/python run_lightrag.py \
                --doc-ids $DOCS \
                --skip-indexing --use-original \
                --query-modes mix \
                --output-dir $EXT/lightrag \
                --questions-dir $EXT/questions \
                > $LOG/${sys}.log 2>&1
            ;;
        hipporag)
            cd $RUN/hipporag
            ./venv/bin/python run_hipporag.py \
                --doc-ids $DOCS \
                --skip-indexing --use-original \
                --output-dir $EXT/hipporag \
                --questions-dir $EXT/questions \
                > $LOG/${sys}.log 2>&1
            ;;
        msgraphrag)
            cd $RUN/msgraphrag
            ./venv/bin/python run_msgraphrag.py \
                --doc-ids $DOCS \
                --skip-indexing --use-original \
                --query-modes local \
                --output-dir $EXT/msgraphrag \
                --questions-dir $EXT/questions \
                > $LOG/${sys}.log 2>&1
            ;;
        raganything)
            cd $RUN/raganything
            # VL_EMBEDDING_* defaults to localhost:9001 / qwen3-vl-embedding (set in runner)
            VLM_BASE_URL="$OPENROUTER_URL" \
            VLM_MODEL="$VL_MODEL" \
            VLM_API_KEY="$OPENROUTER_KEY" \
            $BENCH_PY run_raganything.py \
                --doc-ids $DOCS \
                --skip-indexing --use-original \
                --query-mode mix \
                --output-dir $EXT/raganything_mix \
                --questions-dir $EXT/questions \
                > $LOG/${sys}.log 2>&1
            ;;
        visrag)
            cd $RUN/visrag
            BENCHMARK_PATH_OVERRIDE=$EXT/benchmark.json \
            VLM_BASE_URL="$OPENROUTER_URL" \
            VLM_MODEL="$VL_MODEL" \
            VLM_API_KEY="$OPENROUTER_KEY" \
                $BENCH_PY run_visrag.py \
                    --doc-ids $DOCS \
                    --output-dir $EXT/visrag \
                    --vlm-url "$OPENROUTER_URL" \
                    --vlm-model "$VL_MODEL" \
                    > $LOG/${sys}.log 2>&1
            ;;
        megarag)
            cd $RUN/megarag
            > $LOG/${sys}.log
            for d in $DOCS; do
                echo "[$(date +%H:%M:%S)] [megarag doc_$d]" | tee -a $LOG/${sys}.log
                BENCHMARK_PATH_OVERRIDE=$EXT/benchmark.json \
                VLM_BASE_URL="$OPENROUTER_URL" \
                VLM_MODEL="$VL_MODEL" \
                VLM_API_KEY="$OPENROUTER_KEY" \
                    ./venv/bin/python run_megarag.py \
                        --doc-ids $d \
                        --skip-indexing --query-mode mix \
                        --output-dir $EXT/megarag \
                        --eval-output-dir $EXT/megarag/eval_format \
                        >> $LOG/${sys}.log 2>&1
            done
            ;;
        *)
            echo "Unknown system: $sys"
            return 1
            ;;
    esac

    local dt=$(($(date +%s) - s_start))
    echo "[$(date +%H:%M:%S)] FINISHED $sys (${dt}s)"
}

for s in $SYSTEMS; do
    run_one "$s" || echo "WARNING: $s had non-zero exit"
done

echo
echo "============================================================"
echo "ALL DONE in $(($(date +%s) - START))s"
echo "============================================================"
for sys in $SYSTEMS; do
    if [ -d $EXT/$sys ]; then
        n=$(find $EXT/$sys -name "doc_*.json" -size +1k 2>/dev/null | wc -l)
        echo "  $sys: $n result files in $EXT/$sys"
    fi
    if [ -d $EXT/${sys}_mix ]; then
        n=$(find $EXT/${sys}_mix -name "doc_*.json" -size +1k 2>/dev/null | wc -l)
        echo "  ${sys}_mix: $n result files in $EXT/${sys}_mix"
    fi
done
