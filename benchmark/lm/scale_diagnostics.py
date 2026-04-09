"""
Scale diagnostics: per-parameter analysis of optimizer behavior at 300M scale.

Runs V3 and the working Dion beta=0.3 side-by-side for N steps,
logging per-parameter, per-step internals to understand the dynamics.

Tracks:
  - Per-param: ||G||, ||M||, ||R||, ||D||, ||update||
  - Per-param: R/G ratio, M/G ratio
  - Per-param: effective rank of W, column norm spectrum
  - Per-param: nu_t, delta_t, epsilon_hat
  - Per-param: per-mode beta values, per-mode rho
  - Per-param: lr_scale, actual step size
  - Global: loss, grad_norm, total comm
  - Shape analysis: how do large vs small matrices differ?
  - Temporal: how do quantities evolve over steps?

The goal is NOT to fix V3 — it's to understand the failure mode
deeply enough to redesign the algorithm correctly.
"""

import json
import math
import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from benchmark.lm.llama import LLaMA
from benchmark.lm.data import load_wikitext103, create_dataloaders
from benchmark.adadion_v3.adadion_v3 import _partial_orth, _col_norm, _orth, _effective_rank


def run_diagnostics(
    max_steps: int = 500,
    device: str = "cuda",
    output_dir: str = "results/scale_diagnostics",
    seed: int = 42,
):
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # Model
    model = LLaMA(dim=1024, n_layers=20, n_heads=16, ffn_dim=2816).to(device)
    print(f"LLaMA: {sum(p.numel() for p in set(model.parameters()))/1e6:.1f}M params")

    # Data
    train_ds, val_ds = load_wikitext103(seq_len=1024)
    train_loader, _ = create_dataloaders(train_ds, val_ds, batch_size=4)

    # Identify matrix params
    param_info = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "tok_emb" not in name and "lm_head" not in name:
            param_info.append({"name": name, "shape": tuple(p.shape), "numel": p.numel()})

    print(f"{len(param_info)} matrix params")
    print(f"Shapes: {set(pi['shape'] for pi in param_info)}")

    # Run both methods side-by-side using manual optimizer logic
    methods = {
        # === Baselines ===
        "dion_beta1":  {"beta": 1.0, "right_factor": "colnorm", "per_mode": False, "lr_scale_mode": "dion"},
        "dion_beta03": {"beta": 0.3, "right_factor": "colnorm", "per_mode": False, "lr_scale_mode": "dion"},
        "dion_beta01": {"beta": 0.1, "right_factor": "colnorm", "per_mode": False, "lr_scale_mode": "dion"},
        # === V3 with CORRECT Dion LR scaling (isolate each innovation) ===
        "v3_partial_orth_beta03":  {"beta": 0.3, "right_factor": "partial_orth", "per_mode": False, "lr_scale_mode": "dion"},
        "v3_partial_orth_beta01":  {"beta": 0.1, "right_factor": "partial_orth", "per_mode": False, "lr_scale_mode": "dion"},
        "v3_permode_colnorm":      {"beta": 0.1, "right_factor": "colnorm", "per_mode": True, "lr_scale_mode": "dion"},
        "v3_full":                 {"beta": 0.1, "right_factor": "partial_orth", "per_mode": True, "lr_scale_mode": "dion"},
    }

    all_logs = {name: [] for name in methods}

    for method_name, cfg in methods.items():
        print(f"\n{'='*60}")
        print(f"Running: {method_name} (beta={cfg['beta']}, rf={cfg['right_factor']}, "
              f"pm={cfg['per_mode']}, lr_scale={cfg['lr_scale_mode']})")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        model_copy = LLaMA(dim=1024, n_layers=20, n_heads=16, ffn_dim=2816).to(device)
        model_copy.load_state_dict(model.state_dict())  # same init

        # Scalar params
        scalar_params = [p for n, p in model_copy.named_parameters()
                         if p.ndim != 2 or "tok_emb" in n or "lm_head" in n]
        adamw = AdamW([{"params": scalar_params, "lr": 3e-4, "weight_decay": 0.1}], lr=3e-4)

        # Manual Dion state per matrix param
        mat_params = []
        states = {}
        for name, p in model_copy.named_parameters():
            if p.ndim == 2 and "tok_emb" not in name and "lm_head" not in name:
                mat_params.append((name, p))
                m, n = p.shape
                r = min(64, m, n)
                states[name] = {
                    "R": torch.zeros_like(p),
                    "V": _orth(torch.randn(n, r, device=device, dtype=p.dtype)),
                    "rho_ema": torch.zeros(r, device=device),
                    "prev_modes": None,
                    "rank": r,
                }

        model_copy.train()
        train_iter = iter(train_loader)
        base_lr = 0.01

        for step in range(max_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device), y.to(device)

            # LR schedule
            warmup = 200
            if step < warmup:
                lr = base_lr * step / warmup
            else:
                progress = (step - warmup) / max(max_steps - warmup, 1)
                lr = base_lr * 0.1 + base_lr * 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

            logits = model_copy(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            total_gnorm = torch.nn.utils.clip_grad_norm_(model_copy.parameters(), 1.0).item()

            step_data = {
                "step": step, "loss": loss.item(), "lr": lr, "grad_norm": total_gnorm,
                "params": {},
            }

            # Manual update for each matrix param
            with torch.no_grad():
                for name, p in mat_params:
                    if p.grad is None:
                        continue
                    G = p.grad
                    st = states[name]
                    m, n = G.shape
                    r = st["rank"]
                    R = st["R"]
                    beta = cfg["beta"]

                    # Weight decay
                    p.mul_(1 - lr * 0.1)

                    # Buffer
                    M = G + R

                    # Power iteration
                    V_prev = st["V"]
                    U = _orth(M @ V_prev)
                    W = M.t() @ U

                    # Right factor
                    if cfg["right_factor"] == "colnorm":
                        V_bar = _col_norm(W)
                    else:
                        V_bar = _partial_orth(W, damping=0.5)

                    # Update D
                    D = U @ V_bar.t()

                    # LR scaling
                    if cfg["lr_scale_mode"] == "dion":
                        # Dion convention: scale by mean column norm of W
                        # (this implicitly accounts for gradient magnitude)
                        col_norms = W.norm(dim=0)
                        lr_scale = col_norms.mean().item() / math.sqrt(r)
                    elif cfg["lr_scale_mode"] == "sqrt_max":
                        lr_scale = math.sqrt(max(m, n) / r)
                    elif cfg["lr_scale_mode"] == "none":
                        lr_scale = 1.0

                    actual_lr = lr * lr_scale
                    p.add_(D, alpha=-actual_lr)

                    # Per-mode persistence
                    UtG = U.t() @ G
                    if cfg["per_mode"] and st["prev_modes"] is not None and st["prev_modes"].shape == UtG.shape:
                        cos_sim = F.cosine_similarity(UtG, st["prev_modes"], dim=1)
                        st["rho_ema"] = 0.95 * st["rho_ema"] + 0.05 * cos_sim
                    st["prev_modes"] = UtG.clone()

                    # Per-mode beta
                    if cfg["per_mode"] and step > 1:
                        b = torch.sigmoid(10.0 * st["rho_ema"])
                        b = b * (beta / b.mean().clamp(min=0.01))
                        b = b.clamp(0.05, 0.95)
                    else:
                        b = torch.full((r,), beta, device=device)

                    # Error feedback
                    UtM = U.t() @ M
                    captured = U @ UtM
                    missed = M - captured
                    retained = U @ ((1 - b).unsqueeze(1) * UtM)
                    R_new = missed + retained

                    # R clamping
                    G_norm = G.norm().item()
                    R_new_norm = R_new.norm().item()
                    clamped = False
                    if G_norm > 1e-8 and R_new_norm / G_norm > 2.0:
                        R_new = R_new * (2.0 * G_norm / R_new_norm)
                        R_new_norm = R_new.norm().item()
                        clamped = True

                    st["R"] = R_new
                    st["V"] = _partial_orth(W, damping=0.5) if cfg["right_factor"] == "partial_orth" else _col_norm(W)

                    # --- DIAGNOSTICS (every step for first 50, then every 50) ---
                    if step < 50 or step % 50 == 0:
                        col_norms = W.norm(dim=0)
                        D_norm = D.norm().item()
                        M_norm = M.norm().item()
                        eps_hat = missed.norm().item() / max(M_norm, 1e-12)

                        # nu_t
                        sv_V = torch.linalg.svdvals(V_bar)
                        nu_t = sv_V[0].item()

                        # Effective rank
                        erank = _effective_rank(col_norms)

                        # Per-mode beta stats
                        b_mean = b.mean().item()
                        b_std = b.std().item()

                        pdata = {
                            "G_norm": G_norm,
                            "M_norm": M_norm,
                            "R_norm": R_new_norm,
                            "R_over_G": R_new_norm / max(G_norm, 1e-12),
                            "D_norm": D_norm,
                            "update_norm": actual_lr * D_norm,
                            "lr_scale": lr_scale,
                            "actual_lr": actual_lr,
                            "nu_t": nu_t,
                            "eps_hat": eps_hat,
                            "erank": erank,
                            "col_norm_max": col_norms.max().item(),
                            "col_norm_min": col_norms.min().item(),
                            "col_norm_ratio": col_norms.max().item() / max(col_norms.min().item(), 1e-12),
                            "beta_mean": b_mean,
                            "beta_std": b_std,
                            "clamped": clamped,
                            "shape": [m, n],
                        }
                        step_data["params"][name] = pdata

            # AdamW for scalar params
            adamw.step()
            for p_all in model_copy.parameters():
                if p_all.grad is not None:
                    p_all.grad = None

            if step < 50 or step % 50 == 0:
                all_logs[method_name].append(step_data)

            if step % 100 == 0:
                print(f"  step {step:4d} | loss {loss.item():.4f} | gnorm {total_gnorm:.3f} | lr {lr:.5f}")

    # Save
    outfile = os.path.join(output_dir, "diagnostics.json")
    with open(outfile, "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"\nSaved to {outfile}")

    # Analysis
    print(f"\n{'='*80}")
    print("DIAGNOSTIC ANALYSIS")
    print(f"{'='*80}")

    for method_name in methods:
        logs = all_logs[method_name]
        if not logs:
            continue
        late = [l for l in logs if l["step"] >= max_steps * 0.5]
        if not late:
            late = logs[-5:]

        print(f"\n--- {method_name} ---")
        print(f"  Final loss: {logs[-1]['loss']:.4f}")

        # Aggregate across params
        for key in ["R_over_G", "lr_scale", "update_norm", "nu_t", "eps_hat",
                     "erank", "col_norm_ratio", "beta_mean", "G_norm"]:
            vals = []
            for l in late:
                for pdata in l["params"].values():
                    if key in pdata:
                        vals.append(pdata[key])
            if vals:
                print(f"  {key:20s}: mean={sum(vals)/len(vals):.4f} "
                      f"min={min(vals):.4f} max={max(vals):.4f}")

        # Check clamping frequency
        clamp_count = sum(
            1 for l in logs for pd in l["params"].values() if pd.get("clamped", False)
        )
        total_count = sum(len(l["params"]) for l in logs)
        print(f"  R clamped: {clamp_count}/{total_count} ({100*clamp_count/max(total_count,1):.1f}%)")

        # Shape-specific analysis
        by_shape = defaultdict(list)
        for l in late:
            for name, pd in l["params"].items():
                shape_key = f"{pd['shape'][0]}x{pd['shape'][1]}"
                by_shape[shape_key].append(pd)

        print(f"  By shape:")
        for shape_key, pds in sorted(by_shape.items()):
            avg_rog = sum(pd["R_over_G"] for pd in pds) / len(pds)
            avg_lr = sum(pd["lr_scale"] for pd in pds) / len(pds)
            avg_upd = sum(pd["update_norm"] for pd in pds) / len(pds)
            avg_gnorm = sum(pd["G_norm"] for pd in pds) / len(pds)
            print(f"    {shape_key:>12s}: R/G={avg_rog:.2f} lr_s={avg_lr:.2f} "
                  f"upd={avg_upd:.4f} G={avg_gnorm:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="results/scale_diagnostics")
    args = parser.parse_args()
    run_diagnostics(max_steps=args.max_steps, device=args.device, output_dir=args.output_dir)
