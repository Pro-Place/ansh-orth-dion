#!/bin/bash
# NeurIPS experiments on 4x A40
set -euo pipefail
cd /workspace/adadion-cifar/benchmark

echo "=== NeurIPS Experiment Suite ==="
echo "Started: $(date)"

run() {
    local gpu=$1 depth=$2 width=$3 opt=$4 dataset=$5 outdir=$6
    local name="wrn${depth}_${width}_${opt}_${dataset}"
    if [ -f "$outdir/$name/summary.json" ]; then
        echo "[GPU $gpu] SKIP $name (done)"
        return
    fi
    echo "[GPU $gpu] START $name $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES=$gpu python3 wide_resnet_scaling.py \
        --optimizer $opt --depth $depth --width $width \
        --dataset $dataset --epochs 100 --seed 42 --gpu 0 \
        --output_dir "$outdir" 2>&1 | grep -E 'FINISHED' | sed "s/^/[GPU $gpu] /"
    echo "[GPU $gpu] DONE $name $(date +%H:%M:%S)"
}

# ── PHASE 1: CIFAR-100 Wide ResNet scaling ──
echo ""
echo "=== PHASE 1: CIFAR-100 WideResNet (12 runs on 4 GPUs) ==="
RD="./results/cifar100_scaling"
mkdir -p "$RD"

# GPU 0: AdamW all widths + WRN-28-10 ResNet baseline
(
    run 0 28 2 adamw cifar100 "$RD"
    run 0 28 4 adamw cifar100 "$RD"
    run 0 28 10 adamw cifar100 "$RD"
) &

# GPU 1: Dion all widths
(
    run 1 28 2 dion cifar100 "$RD"
    run 1 28 4 dion cifar100 "$RD"
    run 1 28 10 dion cifar100 "$RD"
) &

# GPU 2: AdaDion all widths
(
    run 2 28 2 adadion cifar100 "$RD"
    run 2 28 4 adadion cifar100 "$RD"
    run 2 28 10 adadion cifar100 "$RD"
) &

# GPU 3: Muon all widths
(
    run 3 28 2 muon cifar100 "$RD"
    run 3 28 4 muon cifar100 "$RD"
    run 3 28 10 muon cifar100 "$RD"
) &

wait
echo "=== PHASE 1 DONE: $(date) ==="

# ── PHASE 2: DDP timing ──
echo ""
echo "=== PHASE 2: DDP Timing (4-GPU WRN-28-10) ==="
torchrun --nproc_per_node=4 distributed_benchmark.py --output_dir ./results/ddp_wrn 2>&1 | grep -E 'ms/step|MB/step|compression'

echo ""
echo "=== ALL DONE: $(date) ==="
