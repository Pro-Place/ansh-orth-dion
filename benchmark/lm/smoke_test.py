"""
Quick smoke test for all Dion variants on synthetic data.
Verifies: model forward/backward, optimizer step, diagnostics collection.
No WikiText download needed.
"""

import torch
import sys
sys.path.insert(0, ".")

from benchmark.lm.model import SmallGPT2
from benchmark.lm.dion_variants import create_dion_variant, VARIANT_REGISTRY, DionBase
from torch.optim import AdamW
import torch.nn.functional as F


def smoke_test(device="cpu", steps=5, seq_len=64, batch_size=2):
    print(f"Smoke test on {device}, {steps} steps\n")

    # Small model for speed
    model = SmallGPT2(
        vocab_size=1000, max_seq_len=seq_len,
        hidden=128, n_heads=4, n_layers=2, tie_weights=True,
    ).to(device)
    print(f"Model: {sum(p.numel() for p in set(model.parameters()))/1e6:.2f}M params\n")

    # Separate matrix vs non-matrix params
    matrix_params = []
    scalar_params = []
    for name, p in model.named_parameters():
        if p.ndim == 2 and "wte" not in name and "wpe" not in name and "lm_head" not in name:
            matrix_params.append(p)
        else:
            scalar_params.append(p)

    print(f"Matrix params: {len(matrix_params)}, Scalar params: {len(scalar_params)}\n")

    # Test each variant
    results = {}
    for variant_name in VARIANT_REGISTRY:
        print(f"--- {variant_name} ---")
        # Reset model
        model_copy = SmallGPT2(
            vocab_size=1000, max_seq_len=seq_len,
            hidden=128, n_heads=4, n_layers=2, tie_weights=True,
        ).to(device)

        mat_params = [p for name, p in model_copy.named_parameters()
                      if p.ndim == 2 and "wte" not in name and "wpe" not in name and "lm_head" not in name]
        scl_params = [p for name, p in model_copy.named_parameters()
                      if p not in set(mat_params)]

        extra = {}
        if variant_name == "soft_isometry":
            extra = {"s_mode": "normalized", "s_floor": 0.1}
        elif variant_name == "modewise_beta":
            extra = {"b_mode": "per_mode", "use_orth": True}
        elif variant_name == "pa_dion":
            extra = {"beta_min": 0.0, "beta_max": 1.0}
        elif variant_name == "rnorm_dion":
            extra = {"R_target": 1.8, "k_p": 0.3}
        elif variant_name == "reentry_dion":
            extra = {"reentry_target": 0.3, "k_p": 0.5}
        elif variant_name == "polyak_dion":
            extra = {}

        base_kwargs = dict(lr=0.01, rank=8, mu=0.0, collect_diagnostics=True)
        if variant_name != "polyak_dion":
            base_kwargs["beta"] = 0.3
        opt = create_dion_variant(
            variant_name, mat_params, **base_kwargs, **extra
        )
        adamw = AdamW(scl_params, lr=0.01) if scl_params else None

        losses = []
        for step in range(steps):
            x = torch.randint(0, 1000, (batch_size, seq_len), device=device)
            y = torch.randint(0, 1000, (batch_size, seq_len), device=device)

            logits = model_copy(x)
            loss = F.cross_entropy(logits.view(-1, 1000), y.view(-1))
            loss.backward()

            opt.step()
            if adamw:
                adamw.step()
            opt.zero_grad(set_to_none=True)
            if adamw:
                adamw.zero_grad(set_to_none=True)

            losses.append(loss.item())

        # Check diagnostics
        diag = opt.get_diagnostics() if isinstance(opt, DionBase) else {}
        diag_keys = set()
        for d in diag.values():
            diag_keys.update(d.keys())

        loss_start = losses[0]
        loss_end = losses[-1]
        ok = loss_end < loss_start + 5.0  # allow some variance on random data

        print(f"  Loss: {loss_start:.3f} -> {loss_end:.3f} {'OK' if ok else 'WARN'}")
        print(f"  Diag keys: {len(diag_keys)} metrics")
        if diag:
            d0 = next(iter(diag.values()))
            for key in ["R_norm", "epsilon_hat", "nu_t", "rho_t", "ReEntryNorm",
                        "T_star_norm", "E_norm", "S_norm", "q_t", "u_t", "s_t",
                        "shadowing_error", "shadowing_bound"]:
                if key in d0:
                    print(f"    {key}: {d0[key]:.4f}")
        print()

        results[variant_name] = {"ok": ok, "loss_start": loss_start, "loss_end": loss_end}

    # Summary
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    all_ok = True
    for name, r in results.items():
        status = "PASS" if r["ok"] else "FAIL"
        if not r["ok"]:
            all_ok = False
        print(f"  {name:<25} {status}  {r['loss_start']:.3f} -> {r['loss_end']:.3f}")

    print(f"\n{'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
    return all_ok


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    smoke_test(device=device, steps=5)
