#!/usr/bin/env python3
"""
Stage 10: Asymmetric WhiteroomTransformer Training (Parallel)

Trains all 5 seeds simultaneously using a shared data generation pool.

Architecture:
- Encoder (1 layer, block-diagonal masked): isolates A and B structurally
- Decoder (5 layers, free): carries the composition load

Same training protocol as Stage 5/9 (2-phase plateau-detected curriculum).
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from _agent.scripts.stage5.train_stage5 import SharedDataServer
from _agent.scripts.stage10.train_stage10 import train_stage10


def _seed_worker(seed, outdir, kwargs, comp_q, attr_q, curr_q):
    """Entry point for each per-seed training process."""
    import torch
    torch.manual_seed(seed)

    checkpoint_dir = str(Path(outdir) / f"stage10-seed{seed}")
    train_stage10(
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        comp_queue=comp_q,
        attr_queue=attr_q,
        curr_queue=curr_q,
        **kwargs,
    )


def train_stage10_parallel(
    outdir: str,
    seeds: str = "1,2,3,4,5",
    n_workers: int = 16,
    balance_archetypes: bool = False,
    cooccurrence_damp: float = 0.7,
    curriculum_prob: float = 0.4,
    valid_weight: float = 1.0,
    d_model: int = 64,
    nhead: int = 4,
    dim_feedforward: int = 256,
    max_steps: int = 200_000,
    num_encoder_layers: int = 1,
    num_decoder_layers: int = 5,
    log_every: int = 500,
    checkpoint_every: int = 10_000,
    block_diag_mask: bool = True,
    warmup_steps: int = 0,
    plateau_window: int = 10,
    plateau_threshold: float = 5e-5,
    min_phase_steps: int = 10_000,
):
    seed_list = [int(s) for s in seeds.split(",")]
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"Stage 10: Asymmetric WhiteroomTransformer (d={d_model} enc={num_encoder_layers} dec={num_decoder_layers})")
    print(f"Output directory: {outdir}")
    print(f"Seeds: {seed_list}")
    print("=" * 80)

    server = SharedDataServer(
        base_seed=42,
        n_workers=n_workers,
        queue_size=4096,
        balance_archetypes=balance_archetypes,
        cooccurrence_damp=cooccurrence_damp,
    )
    print(f"SharedDataServer started: {n_workers} workers, queues ready")

    train_kwargs = dict(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=0.1,
        lr=3e-4,
        batch_size=64,
        max_steps=max_steps,
        curriculum_prob=curriculum_prob,
        valid_weight=valid_weight,
        block_diag_mask=block_diag_mask,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        warmup_steps=warmup_steps,
        plateau_window=plateau_window,
        plateau_threshold=plateau_threshold,
        min_phase_steps=min_phase_steps,
        balance_archetypes=balance_archetypes,
        cooccurrence_damp=cooccurrence_damp,
    )

    procs = []
    for seed in seed_list:
        ckpt_dir = Path(outdir) / f"stage10-seed{seed}"
        if (ckpt_dir / "checkpoint_final.pt").exists():
            print(f"[seed {seed}] already done, skipping")
            continue

        p = mp.Process(
            target=_seed_worker,
            args=(seed, outdir, train_kwargs,
                  server.comp_q, server.attr_q, server.curr_q),
            name=f"stage10-seed{seed}",
        )
        p.start()
        print(f"[seed {seed}] started PID {p.pid}")
        procs.append((seed, p))

    for seed, p in procs:
        p.join()
        print(f"[seed {seed}] finished (exit code {p.exitcode})")

    server.stop()
    print("\nAll Stage 10 seeds complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 10: Asymmetric Transformer Training (Parallel)")
    parser.add_argument("--outdir",             type=str,   required=True)
    parser.add_argument("--seeds",              type=str,   default="1,2,3,4,5")
    parser.add_argument("--n-workers",          type=int,   default=16)
    parser.add_argument("--balance-archetypes", action="store_true")
    parser.add_argument("--cooccurrence-damp",  type=float, default=0.7)
    parser.add_argument("--curriculum-prob",    type=float, default=0.4)
    parser.add_argument("--valid-weight",       type=float, default=1.0)
    parser.add_argument("--d-model",            type=int,   default=64)
    parser.add_argument("--nhead",              type=int,   default=4)
    parser.add_argument("--ffn-dim",            type=int,   default=256)
    parser.add_argument("--max-steps",          type=int,   default=200_000)
    parser.add_argument("--enc-layers",         type=int,   default=1)
    parser.add_argument("--dec-layers",         type=int,   default=5)
    parser.add_argument("--log-every",          type=int,   default=500)
    parser.add_argument("--checkpoint-every",   type=int,   default=10_000)
    parser.add_argument("--no-block-diag",        action="store_true")
    parser.add_argument("--warmup-steps",        type=int,   default=0)
    parser.add_argument("--plateau-window",      type=int,   default=10)
    parser.add_argument("--plateau-threshold",   type=float, default=5e-5)
    parser.add_argument("--min-phase-steps",     type=int,   default=10_000)

    args = parser.parse_args()
    train_stage10_parallel(
        outdir=args.outdir,
        seeds=args.seeds,
        n_workers=args.n_workers,
        balance_archetypes=args.balance_archetypes,
        cooccurrence_damp=args.cooccurrence_damp,
        curriculum_prob=args.curriculum_prob,
        max_steps=args.max_steps,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.ffn_dim,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        valid_weight=args.valid_weight,
        block_diag_mask=not args.no_block_diag,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        plateau_window=args.plateau_window,
        plateau_threshold=args.plateau_threshold,
        min_phase_steps=args.min_phase_steps,
    )
