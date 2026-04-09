"""
Response-surface experiment for buffer scaling analysis.

Freeze a checkpoint and scale the live buffer by lambda:
  M(lambda) = G + lambda * R

Measure one-step loss for each lambda. Fit:
  delta_f ≈ a*lambda - b*lambda^2 - c*lambda^3

Then a/(2b) gives first approximation to the R_norm sweet spot.

Identifies the optimal buffer-to-gradient ratio.
"""

import argparse
import json
import math
import os
import sys
import copy

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.lm.model import SmallGPT2, create_small_gpt2
from benchmark.lm.data import load_wikitext103, create_dataloaders
from benchmark.lm.dion_variants import _col_norm, _orth, _col_norms


def response_surface_at_checkpoint(
    checkpoint_path: str,
    lambdas: list = None,
    n_batches: int = 10,
    device: str = "cuda",
    output_dir: str = "results/lm/response_surface",
):
    """
    For each lambda:
      1. Load checkpoint (model weights + optimizer state with R buffer)
      2. Get a batch, compute gradient G
      3. Form M(lambda) = G + lambda * R (using R from optimizer state)
      4. Do one rank-r power iteration + ColNorm step
      5. Apply update, measure new loss
      6. Record delta_f = new_loss - old_loss
    """
    if lambdas is None:
        lambdas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Rebuild model
    model_cfg = ckpt.get("config", None)
    if model_cfg and hasattr(model_cfg, "model"):
        model = create_small_gpt2(**model_cfg.model.__dict__).to(device)
    else:
        model = create_small_gpt2().to(device)
    model.load_state_dict(ckpt["model_state"])

    # Extract R buffers from optimizer state
    dion_state = ckpt["dion_state"]
    R_buffers = {}
    V_buffers = {}
    for param_key, state in dion_state["state"].items():
        if "R" in state:
            R_buffers[param_key] = state["R"].to(device)
        if "V" in state:
            V_buffers[param_key] = state["V"].to(device)

    # Map param indices to actual parameters
    param_list = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "wte" not in name and "wpe" not in name and "lm_head" not in name:
            param_list.append((name, p))

    print(f"Found {len(R_buffers)} R buffers, {len(param_list)} matrix params")

    # Load data
    train_ds, val_ds = load_wikitext103(seq_len=1024)
    train_loader, _ = create_dataloaders(train_ds, val_ds, batch_size=8)

    # Get multiple batches for averaging
    batches = []
    loader_iter = iter(train_loader)
    for _ in range(n_batches):
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)
        batches.append((x.to(device), y.to(device)))

    results = []
    rank = 64

    for lam in lambdas:
        delta_fs = []

        for batch_idx, (x, y) in enumerate(batches):
            # Reload fresh model weights
            model.load_state_dict(ckpt["model_state"])
            model.train()

            # Forward to get loss_0
            with torch.no_grad():
                logits_0 = model(x)
                loss_0 = F.cross_entropy(logits_0.view(-1, logits_0.size(-1)), y.view(-1)).item()

            # Compute gradient
            model.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()

            # Apply one Dion step with M(lambda) = G + lambda * R
            with torch.no_grad():
                for idx, (name, p) in enumerate(param_list):
                    if p.grad is None:
                        continue
                    G = p.grad
                    R = R_buffers.get(idx, torch.zeros_like(G))
                    V = V_buffers.get(idx, None)

                    m, n = G.shape
                    r = min(rank, m, n)

                    # M(lambda) = G + lambda * R
                    M = G + lam * R

                    # Power iteration
                    if V is not None and V.shape[1] == r:
                        V_warm = V
                    else:
                        V_warm = torch.randn(n, r, device=device, dtype=G.dtype)
                        V_warm = _orth(V_warm)

                    U = _orth(M @ V_warm)
                    W = M.t() @ U
                    V_bar = _col_norm(W)
                    D = U @ V_bar.t()

                    # Apply update (lr from checkpoint config)
                    lr = 0.01
                    p.add_(D, alpha=-lr)

            # Measure new loss
            with torch.no_grad():
                logits_new = model(x)
                loss_new = F.cross_entropy(
                    logits_new.view(-1, logits_new.size(-1)), y.view(-1)
                ).item()

            delta_fs.append(loss_new - loss_0)

        mean_delta = sum(delta_fs) / len(delta_fs)
        std_delta = (sum((d - mean_delta)**2 for d in delta_fs) / max(len(delta_fs)-1, 1))**0.5

        results.append({
            "lambda": lam,
            "delta_f_mean": mean_delta,
            "delta_f_std": std_delta,
            "n_batches": len(delta_fs),
        })
        print(f"  lambda={lam:5.2f}: delta_f = {mean_delta:+.6f} ± {std_delta:.6f}")

    # Fit cubic: delta_f ≈ c1*lam + c2*lam^2 + c3*lam^3
    import numpy as np
    lams = np.array([r["lambda"] for r in results])
    deltas = np.array([r["delta_f_mean"] for r in results])

    # Fit polynomial (degree 3, no intercept since delta_f(0) should be ~0)
    # delta_f = a*lam + b*lam^2 + c*lam^3
    A = np.column_stack([lams, lams**2, lams**3])
    coeffs, residuals, _, _ = np.linalg.lstsq(A, deltas, rcond=None)
    a, b, c = coeffs

    # Sweet spot: minimize delta_f → d/dlam (a*lam + b*lam^2 + c*lam^3) = 0
    # a + 2b*lam + 3c*lam^2 = 0
    # For the quadratic approximation (ignoring cubic): sweet spot ≈ -a/(2b)
    if abs(b) > 1e-12:
        sweet_spot_quadratic = -a / (2 * b)
    else:
        sweet_spot_quadratic = float("nan")

    # Full cubic roots
    discriminant = 4*b**2 - 12*a*c
    if discriminant >= 0 and abs(c) > 1e-12:
        root1 = (-2*b + math.sqrt(discriminant)) / (6*c)
        root2 = (-2*b - math.sqrt(discriminant)) / (6*c)
        # Pick the positive root that's a minimum
        sweet_spot_cubic = min(
            (r for r in [root1, root2] if r > 0),
            default=sweet_spot_quadratic
        )
    else:
        sweet_spot_cubic = sweet_spot_quadratic

    fit_result = {
        "a": float(a), "b": float(b), "c": float(c),
        "sweet_spot_quadratic": float(sweet_spot_quadratic),
        "sweet_spot_cubic": float(sweet_spot_cubic),
    }

    print(f"\nFit: delta_f ≈ {a:.6f}*λ + {b:.6f}*λ² + {c:.6f}*λ³")
    print(f"Sweet spot (quadratic): λ* ≈ {sweet_spot_quadratic:.3f}")
    print(f"Sweet spot (cubic):     λ* ≈ {sweet_spot_cubic:.3f}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    ckpt_name = os.path.basename(os.path.dirname(checkpoint_path))
    step_name = os.path.basename(checkpoint_path).replace(".pt", "")
    outfile = os.path.join(output_dir, f"response_surface_{ckpt_name}_{step_name}.json")
    with open(outfile, "w") as f:
        json.dump({"results": results, "fit": fit_result, "checkpoint": checkpoint_path}, f, indent=2)
    print(f"Saved to: {outfile}")

    return results, fit_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/lm/response_surface")
    parser.add_argument("--n_batches", type=int, default=10)
    args = parser.parse_args()

    response_surface_at_checkpoint(
        args.checkpoint, device=args.device,
        output_dir=args.output_dir, n_batches=args.n_batches,
    )


if __name__ == "__main__":
    main()
