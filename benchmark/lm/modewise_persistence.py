"""
Per-mode gradient persistence analysis.

Measure persistence and re-entry PER SINGULAR MODE, not just globally.
If modes differ a lot, scalar beta is provably too coarse.

For each mode i (i=1..r):
  - rho_i(t): cosine similarity of mode-i out-of-subspace gradient across steps
  - u_i(t): re-entry coefficient for mode i
  - s_i(t): survival coefficient for mode i
  - energy_i(t): column norm of W (proxy for singular value)

Runs training for N steps, collecting per-mode diagnostics at intervals.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.lm.model import SmallGPT2, create_small_gpt2
from benchmark.lm.data import load_wikitext103, create_dataloaders
from benchmark.lm.dion_variants import _col_norm, _orth, _col_norms, _effective_rank
from benchmark.lm.train import get_lr
from benchmark.lm.configs import TrainConfig


def run_modewise_study(
    max_steps: int = 10000,
    rank: int = 64,
    beta: float = 0.3,
    diag_interval: int = 100,
    device: str = "cuda",
    output_dir: str = "results/lm/modewise_persistence",
    seed: int = 42,
):
    torch.manual_seed(seed)
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    os.makedirs(output_dir, exist_ok=True)

    model = create_small_gpt2().to(device)
    train_ds, val_ds = load_wikitext103(seq_len=1024)
    train_loader, val_loader = create_dataloaders(train_ds, val_ds, batch_size=8)

    # Get matrix params
    matrix_params = []
    matrix_names = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "wte" not in name and "wpe" not in name and "lm_head" not in name:
            matrix_params.append(p)
            matrix_names.append(name)

    # Scalar params
    scalar_params = [p for name, p in model.named_parameters()
                     if p.requires_grad and p not in set(matrix_params)]
    adamw = AdamW([{"params": scalar_params, "lr": 3e-4, "weight_decay": 0.1}], lr=3e-4)

    # Manual Dion state
    states = {}
    for i, p in enumerate(matrix_params):
        m, n = p.shape
        r = min(rank, m, n)
        states[i] = {
            "R": torch.zeros_like(p),
            "V": _orth(torch.randn(n, r, device=device, dtype=p.dtype)),
            "momentum": torch.zeros_like(p),
            # Per-mode tracking
            "prev_mode_oos": None,          # (r, n) previous out-of-subspace per mode
            "mode_rho_ema": torch.zeros(r, device=device),
        }

    cfg = TrainConfig(max_steps=max_steps, warmup_steps=3000)
    model.train()
    train_iter = iter(train_loader)
    start_time = time.time()

    all_diagnostics = []  # list of per-step dicts

    for step in range(max_steps):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        x, y = x.to(device), y.to(device)

        lr = get_lr(step, cfg)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        collect = (step % diag_interval == 0)
        step_diag = {"step": step, "train_loss": loss.item(), "lr": lr} if collect else None

        # Manual Dion step for each matrix param
        with torch.no_grad():
            for i, p in enumerate(matrix_params):
                if p.grad is None:
                    continue
                G = p.grad
                st = states[i]
                m, n = G.shape
                r = min(rank, m, n)

                # Momentum
                st["momentum"].copy_(G)

                # Buffer
                M = G + st["R"]

                # Power iteration
                U = _orth(M @ st["V"])
                W = M.t() @ U  # (n x r)

                if collect and i == 0:  # detailed diagnostics for first param
                    col_norms = _col_norms(W)
                    sorted_norms = col_norms.sort(descending=True).values

                    # --- Per-mode diagnostics ---
                    # U^T @ G gives per-mode gradient projection
                    UtG = U.t() @ G  # (r, n)
                    # Out-of-subspace gradient per mode
                    out_G = G - U @ UtG  # (m, n) global
                    # Per-mode: project out_G onto each mode direction
                    # Actually, let's track the per-mode component of M
                    UtM = U.t() @ M  # (r, n) - per mode contribution

                    # Per-mode re-entry: how much of R is captured by each mode
                    UtR = U.t() @ st["R"]  # (r, n)
                    R_mode_norms = UtR.norm(dim=1)  # (r,)
                    R_total = st["R"].norm().item()

                    # Per-mode u_i = ||U_i^T R|| / ||R|| (re-entry per mode)
                    u_per_mode = (R_mode_norms / max(R_total, 1e-12)).tolist()

                    # Per-mode survival: ||(I-P)R|| projected per mode doesn't quite work
                    # Instead: s_i = 1 - u_i^2 (approximate)

                    # Per-mode persistence (rho_i)
                    # Compare UtG at this step with previous step
                    prev = st["prev_mode_oos"]
                    if prev is not None and prev.shape == UtG.shape:
                        # Per-mode cosine similarity
                        cos_per_mode = F.cosine_similarity(UtG, prev, dim=1)  # (r,)
                        rho_per_mode = cos_per_mode.tolist()
                        # EMA
                        st["mode_rho_ema"] = 0.95 * st["mode_rho_ema"] + 0.05 * cos_per_mode
                    else:
                        rho_per_mode = [0.0] * r
                    st["prev_mode_oos"] = UtG.clone()

                    # Global metrics
                    eps_hat = (M - U @ (U.t() @ M)).norm().item() / max(M.norm().item(), 1e-12)
                    V_bar = _col_norm(W)
                    gram = V_bar.t() @ V_bar
                    nu_t = torch.linalg.svdvals(V_bar)[0].item()

                    step_diag["param"] = matrix_names[i]
                    step_diag["epsilon_hat"] = eps_hat
                    step_diag["nu_t"] = nu_t
                    step_diag["R_norm"] = R_total / max(G.norm().item(), 1e-12)
                    step_diag["effective_rank_W"] = _effective_rank(sorted_norms)

                    # Per-mode arrays
                    step_diag["col_norms"] = sorted_norms.tolist()[:10]  # top 10
                    step_diag["rho_per_mode"] = rho_per_mode[:10]        # top 10
                    step_diag["rho_ema_per_mode"] = st["mode_rho_ema"].tolist()[:10]
                    step_diag["u_per_mode"] = u_per_mode[:10]            # top 10

                    # Summary stats across modes
                    rho_tensor = torch.tensor(rho_per_mode)
                    step_diag["rho_mode_mean"] = rho_tensor.mean().item()
                    step_diag["rho_mode_std"] = rho_tensor.std().item()
                    step_diag["rho_mode_min"] = rho_tensor.min().item()
                    step_diag["rho_mode_max"] = rho_tensor.max().item()
                    step_diag["u_mode_mean"] = sum(u_per_mode) / len(u_per_mode)
                    step_diag["u_mode_std"] = torch.tensor(u_per_mode).std().item()

                    # Key question: do modes have very different rho?
                    # If std(rho) >> 0, scalar beta is too coarse
                    step_diag["rho_heterogeneity"] = rho_tensor.std().item() / max(abs(rho_tensor.mean().item()), 1e-6)

                # ColNorm + update
                V_bar = _col_norm(W)
                D = U @ V_bar.t()
                p.mul_(1 - lr * 0.1)  # weight decay
                p.add_(D, alpha=-lr)

                # Error feedback
                P_captured = U @ (U.t() @ M)
                st["R"] = M - beta * P_captured
                st["V"] = _orth(W)

        # AdamW step for scalars
        adamw.step()
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None

        if collect and step_diag is not None:
            all_diagnostics.append(step_diag)

        if step % 1000 == 0:
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.5f}")

    # Save
    outfile = os.path.join(output_dir, f"modewise_r{rank}_beta{beta}.json")
    with open(outfile, "w") as f:
        json.dump(all_diagnostics, f, indent=2)
    print(f"\nSaved {len(all_diagnostics)} diagnostic snapshots to {outfile}")

    # Summary analysis
    if all_diagnostics:
        late = [d for d in all_diagnostics if d["step"] > max_steps * 0.5]
        if late:
            avg_rho_std = sum(d.get("rho_mode_std", 0) for d in late) / len(late)
            avg_rho_mean = sum(d.get("rho_mode_mean", 0) for d in late) / len(late)
            avg_u_std = sum(d.get("u_mode_std", 0) for d in late) / len(late)
            avg_het = sum(d.get("rho_heterogeneity", 0) for d in late) / len(late)

            print(f"\n=== Late-training summary (steps > {max_steps//2}) ===")
            print(f"Mean rho across modes: {avg_rho_mean:.4f}")
            print(f"Std rho across modes:  {avg_rho_std:.4f}")
            print(f"Heterogeneity (std/|mean|): {avg_het:.2f}")
            print(f"Std u_i across modes:  {avg_u_std:.4f}")
            if avg_het > 2.0:
                print(">>> HIGH heterogeneity: scalar beta is likely too coarse!")
            elif avg_het > 1.0:
                print(">>> MODERATE heterogeneity: per-mode beta may help.")
            else:
                print(">>> LOW heterogeneity: scalar beta is probably fine.")

    return all_diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--diag_interval", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/lm/modewise_persistence")
    args = parser.parse_args()

    run_modewise_study(
        max_steps=args.max_steps, rank=args.rank, beta=args.beta,
        diag_interval=args.diag_interval, device=args.device,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
