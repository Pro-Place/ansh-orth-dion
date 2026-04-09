#!/bin/bash
# Full Benchmark Launch Script for RunPod
# Runs the complete benchmark pipeline:
#   1. Smoke test (3 epochs, verify all optimizers work)
#   2. Full benchmark (200 epochs, 2 models × 5 optimizers × 3 seeds)
#   3. LR sweeps for each optimizer
#   4. Analysis and plot generation
#   bash run_benchmark.sh              # Full pipeline
#   bash run_benchmark.sh smoke        # Smoke test only
#   bash run_benchmark.sh full         # Full benchmark only
#   bash run_benchmark.sh sweep        # LR sweeps only
#   bash run_benchmark.sh analyze      # Analysis only

set -euo pipefail

WORK_DIR="${WORK_DIR:-/workspace}"
BENCH_DIR="$WORK_DIR/benchmark"
RESULTS_DIR="$BENCH_DIR/results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Ensure AdaDion V2 is importable
export PYTHONPATH="$WORK_DIR/torchtitan/torchtitan/experiments:${PYTHONPATH:-}"

cd "$BENCH_DIR"
mkdir -p "$RESULTS_DIR"

MODE="${1:-all}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# ── Phase 1: Smoke Test ──────────────────────────────────────────────
run_smoke() {
    log "========================================"
    log "Phase 1: Smoke Test"
    log "========================================"
    python3 cifar10_benchmark.py \
        --mode smoke \
        --output_dir "$RESULTS_DIR/smoke_${TIMESTAMP}" \
        --num_workers 4

    log "Smoke test complete."
}

# ── Phase 2: Full Benchmark ──────────────────────────────────────────
run_full() {
    log "========================================"
    log "Phase 2: Full Benchmark"
    log "  Models: resnet18, vit_small"
    log "  Optimizers: adamw, muon, dion, dion2, adadion"
    log "  Seeds: 42, 123, 456"
    log "  Epochs: 200"
    log "========================================"

    FULL_DIR="$RESULTS_DIR/full_${TIMESTAMP}"
    mkdir -p "$FULL_DIR"

    python3 cifar10_benchmark.py \
        --mode full \
        --output_dir "$FULL_DIR" \
        --num_workers 4

    log "Full benchmark complete. Results in $FULL_DIR"
}

# ── Phase 3: LR Sweeps ──────────────────────────────────────────────
run_sweep() {
    log "========================================"
    log "Phase 3: Learning Rate Sweeps"
    log "========================================"

    SWEEP_DIR="$RESULTS_DIR/sweeps_${TIMESTAMP}"
    mkdir -p "$SWEEP_DIR"

    for OPT in adamw muon dion dion2 adadion; do
        log "  Sweeping $OPT..."
        python3 cifar10_benchmark.py \
            --mode sweep \
            --optimizer "$OPT" \
            --model resnet18 \
            --epochs 100 \
            --output_dir "$SWEEP_DIR/${OPT}" \
            --num_workers 4 || {
                log "  WARNING: $OPT sweep failed, continuing..."
            }
    done

    log "LR sweeps complete. Results in $SWEEP_DIR"
}

# ── Phase 4: Analysis ────────────────────────────────────────────────
run_analyze() {
    log "========================================"
    log "Phase 4: Analysis & Visualization"
    log "========================================"

    # Find the most recent full benchmark results
    LATEST_FULL=$(ls -td "$RESULTS_DIR"/full_* 2>/dev/null | head -1)
    if [ -z "$LATEST_FULL" ]; then
        log "No full benchmark results found, skipping analysis"
        return
    fi

    log "  Analyzing: $LATEST_FULL"
    python3 analysis.py --results_dir "$LATEST_FULL"

    # Also analyze sweeps if available
    LATEST_SWEEP=$(ls -td "$RESULTS_DIR"/sweeps_* 2>/dev/null | head -1)
    if [ -n "$LATEST_SWEEP" ]; then
        for OPT_DIR in "$LATEST_SWEEP"/*/; do
            if [ -d "$OPT_DIR" ]; then
                log "  Analyzing sweep: $OPT_DIR"
                python3 analysis.py --results_dir "$OPT_DIR" || true
            fi
        done
    fi

    log "Analysis complete."
}

# ── Dispatch ─────────────────────────────────────────────────────────

case "$MODE" in
    smoke)
        run_smoke
        ;;
    full)
        run_full
        ;;
    sweep)
        run_sweep
        ;;
    analyze)
        run_analyze
        ;;
    all)
        log "Running full pipeline..."
        log ""
        run_smoke
        log ""
        run_full
        log ""
        run_sweep
        log ""
        run_analyze
        log ""
        log "========================================"
        log "ALL DONE!"
        log "Results: $RESULTS_DIR"
        log "========================================"
        ;;
    *)
        echo "Usage: $0 {smoke|full|sweep|analyze|all}"
        exit 1
        ;;
esac
