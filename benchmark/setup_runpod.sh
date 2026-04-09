#!/bin/bash
# RunPod Environment Setup for CIFAR-10 Optimizer Benchmark

set -euo pipefail

echo "=========================================="
echo "CIFAR-10 Optimizer Benchmark - Setup"
echo "=========================================="

# ── System info ──────────────────────────────────────────────────────
echo ""
echo "[1/7] System information"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ── Clone repo ───────────────────────────────────────────────────────
echo ""
echo "[2/7] Cloning torchtitan (Tatzhiro/replication branch)"
WORK_DIR="${WORK_DIR:-/workspace}"
cd "$WORK_DIR"

if [ ! -d "torchtitan" ]; then
    git clone --branch Tatzhiro/replication --depth 1 \
        https://github.com/Pro-Place/torchtitan.git
    echo "  Cloned successfully"
else
    echo "  torchtitan already exists, pulling latest"
    cd torchtitan && git pull && cd ..
fi

# ── Install Python dependencies ──────────────────────────────────────
echo ""
echo "[3/7] Installing Python dependencies"
pip install --quiet --upgrade pip
pip install --quiet \
    torch torchvision \
    numpy \
    matplotlib \
    seaborn \
    tqdm \
    triton

# ── Install dion package ─────────────────────────────────────────────
echo ""
echo "[4/7] Installing dion optimizer package"
# Try multiple install paths in priority order
if python3 -c "from dion import Dion" 2>/dev/null; then
    echo "  dion already installed"
elif [ -d "torchtitan/third_party/dion" ]; then
    echo "  Installing dion from torchtitan/third_party/dion"
    pip install --quiet -e torchtitan/third_party/dion
elif pip install --quiet dion 2>/dev/null; then
    echo "  Installed dion from PyPI"
elif pip install --quiet dion-optimizer 2>/dev/null; then
    echo "  Installed dion-optimizer from PyPI"
else
    echo "  ERROR: Could not install dion package."
    echo "  Try: pip install git+https://github.com/microsoft/dion.git"
    exit 1
fi

# Verify imports
python3 -c "
import torch, torchvision, numpy, matplotlib
print(f'  torch={torch.__version__}, torchvision={torchvision.__version__}')
print(f'  numpy={numpy.__version__}, matplotlib={matplotlib.__version__}')
try:
    from dion import Dion, Dion2, Muon
    print('  dion package: OK (Dion, Dion2, Muon)')
except ImportError as e:
    print(f'  dion package: MISSING ({e})')
"

# ── Set up benchmark directory ───────────────────────────────────────
echo ""
echo "[5/7] Setting up benchmark directory"
BENCH_DIR="$WORK_DIR/benchmark"

if [ ! -d "$BENCH_DIR" ]; then
    echo "  ERROR: benchmark/ directory not found at $BENCH_DIR"
    echo "  Please copy the benchmark/ directory to $WORK_DIR/"
    echo "  Example: scp -r benchmark/ runpod:$WORK_DIR/"
    exit 1
fi

# Make sure the AdaDion V2 module is importable
export PYTHONPATH="$WORK_DIR/torchtitan/torchtitan/experiments:${PYTHONPATH:-}"
echo "  PYTHONPATH=$PYTHONPATH"

# ── Verify all optimizers load ───────────────────────────────────────
echo ""
echo "[6/7] Verifying optimizer imports"
cd "$BENCH_DIR"
python3 -c "
import sys, os
sys.path.insert(0, '$WORK_DIR/torchtitan/torchtitan/experiments')

print('Testing imports...')
from dion import Dion, Dion2, Muon
print('  Dion:   OK')
print('  Dion2:  OK')
print('  Muon:   OK')

from ortho_matrix.ada_dion_v2.adadion_v2 import AdaDionV2
print('  AdaDionV2: OK')

import torch
print('  AdamW:  OK (torch.optim)')

# Quick sanity: create a small model and each optimizer
import torch.nn as nn
model = nn.Linear(64, 32)
params = list(model.parameters())

# AdamW
opt = torch.optim.AdamW(params, lr=1e-3)
print('  AdamW init: OK')

# Muon (single-GPU, no distributed mesh)
opt = Muon([{'params': [p for p in params if p.ndim >= 2]},
            {'params': [p for p in params if p.ndim < 2], 'algorithm': 'adamw', 'lr': 1e-3}],
           lr=0.02, flatten=True)
print('  Muon init:  OK')

# Dion
opt = Dion([{'params': [p for p in params if p.ndim >= 2]},
            {'params': [p for p in params if p.ndim < 2], 'algorithm': 'adamw', 'lr': 1e-3}],
           lr=0.02, rank_fraction=0.25)
print('  Dion init:  OK')

# Dion2
opt = Dion2([{'params': [p for p in params if p.ndim >= 2]},
             {'params': [p for p in params if p.ndim < 2], 'algorithm': 'adamw', 'lr': 1e-3}],
            lr=0.02, fraction=0.25, flatten=True)
print('  Dion2 init: OK')

print()
print('All optimizers verified successfully!')
" || {
    echo "  WARNING: Some optimizer imports failed. Check error above."
    echo "  The benchmark will skip failed optimizers."
}

# ── Download CIFAR-10 ────────────────────────────────────────────────
echo ""
echo "[7/7] Pre-downloading CIFAR-10 dataset"
cd "$BENCH_DIR"
python3 -c "
from torchvision import datasets
datasets.CIFAR10(root='./data', train=True, download=True)
datasets.CIFAR10(root='./data', train=False, download=True)
print('  CIFAR-10 downloaded')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "To run the benchmark:"
echo "  cd $BENCH_DIR"
echo "  bash run_benchmark.sh"
echo ""
echo "Or run individual experiments:"
echo "  python cifar10_benchmark.py --mode smoke"
echo "  python cifar10_benchmark.py --optimizer muon --model resnet18"
echo "  python cifar10_benchmark.py --mode full"
echo "=========================================="
