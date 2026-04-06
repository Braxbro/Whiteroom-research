#!/usr/bin/env python3
"""
Stage 9: 3-Stage Model Training from Scratch

Parallel launcher for 3-stage model: trains all 5 seeds simultaneously
using a shared data generation pool (SharedDataServer).

Architecture:
- Stage 1 (Encoder): 3 layers, block-diagonal bidirectional attention
- Stage 2 (Adaptation): Linear(64→64) projection to bridge spaces
- Stage 3 (Decoder): 3 layers, free cross-attention

Training: Adaptive freeze curriculum with plateau detection.
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from _agent.scripts.stage5.train_stage5 import SharedDataServer
from _agent.scripts.stage9.train_stage9 import train_stage9


def _seed_worker(seed, outdir, kwargs, comp_q, attr_q, curr_q):
    """Entry point for each per-seed training process."""
    import torch
    torch.manual_seed(seed)

    checkpoint_dir = str(Path(outdir) / f"stage9-seed{seed}")
    train_stage9(
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        comp_queue=comp_q,
        attr_queue=attr_q,
        curr_queue=curr_q,
        **kwargs,
    )


def train_stage9_parallel(
    outdir: str,
    seeds: str = "1,2,3,4,5",
    n_workers: int = 16,
    balance_archetypes: bool = False,
    cooccurrence_damp: float = 0.7,
    curriculum_prob: float = 0.4,
    max_steps: int = 200_000,
    log_every: int = 500,
    checkpoint_every: int = 10_000,
    plateau_window: int = 10,
    plateau_threshold: float = 5e-5,
    min_phase_steps: int = 10_000,
    block_diagonal_encoder: bool = False,
    bidirectional_blocks: bool = False,
):
    """
    Train Stage 9 3-stage model from scratch in parallel.

    All 5 seeds run simultaneously, sharing a data generation pool
    (SharedDataServer with N_WORKERS CPU processes).
    """
    seed_list = [int(s) for s in seeds.split(",")]
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Stage 9: 3-Stage Model from Scratch (Parallel)")
    print(f"Output directory: {outdir}")
    print(f"Seeds: {seed_list}")
    print(f"Bidirectional binding: {bidirectional_blocks or True}")  # always on for Stage 9
    print("=" * 80)

    # Shared data server for all seeds
    server = SharedDataServer(
        base_seed=42,
        n_workers=n_workers,
        queue_size=4096,
        balance_archetypes=balance_archetypes,
        cooccurrence_damp=cooccurrence_damp,
    )
    print(f"SharedDataServer started: {n_workers} workers, queues ready")

    train_kwargs = dict(
        d_model=64,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        lr=3e-4,
        batch_size=64,
        max_steps=max_steps,
        curriculum_prob=curriculum_prob,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        plateau_window=plateau_window,
        plateau_threshold=plateau_threshold,
        min_phase_steps=min_phase_steps,
        balance_archetypes=balance_archetypes,
        cooccurrence_damp=cooccurrence_damp,
        bidir_bind=True,  # Stage 9 always uses bidirectional binding
    )

    # Spawn one process per seed
    procs = []
    for seed in seed_list:
        ckpt_dir = Path(outdir) / f"stage9-seed{seed}"
        if (ckpt_dir / "checkpoint_final.pt").exists():
            print(f"[seed {seed}] already done, skipping")
            continue

        p = mp.Process(
            target=_seed_worker,
            args=(seed, outdir, train_kwargs,
                  server.comp_q, server.attr_q, server.curr_q),
            name=f"stage9-seed{seed}",
        )
        p.start()
        print(f"[seed {seed}] started PID {p.pid}")
        procs.append((seed, p))

    # Wait for all to finish
    for seed, p in procs:
        p.join()
        print(f"[seed {seed}] finished (exit code {p.exitcode})")

    server.stop()
    print("\nAll Stage 9 seeds complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 9: 3-Stage Model Training (Parallel)")
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--n-workers", type=int, default=16)
    parser.add_argument("--balance-archetypes", action="store_true")
    parser.add_argument("--cooccurrence-damp", type=float, default=0.7)
    parser.add_argument("--curriculum-prob", type=float, default=0.4)
    parser.add_argument("--max-steps", type=int, default=200_000)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--plateau-window", type=int, default=10)
    parser.add_argument("--plateau-threshold", type=float, default=5e-5)
    parser.add_argument("--min-phase-steps", type=int, default=10_000)
    parser.add_argument("--block-diagonal-encoder", action="store_true")
    parser.add_argument("--bidirectional-blocks", action="store_true")

    args = parser.parse_args()
    train_stage9_parallel(**vars(args))
