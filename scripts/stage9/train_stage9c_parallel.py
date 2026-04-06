#!/usr/bin/env python3
"""
Stage 9c: 4-Phase Curriculum with Selective Freezing

Parallel launcher for all 5 seeds with shared data generation pool.

Phase sequence (1A → 2A → 1B → 2B):
  1A & 2A: Bridge FROZEN, encoder/decoder free
  1B & 2B: Encoder FROZEN, bridge/decoder free
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from _agent.scripts.stage5.train_stage5 import SharedDataServer
from _agent.scripts.stage9.train_stage9c import train_stage9c


def _seed_worker(seed, outdir, kwargs, comp_q, attr_q, curr_q):
    """Entry point for each per-seed training process."""
    import torch
    torch.manual_seed(seed)

    checkpoint_dir = str(Path(outdir) / f"stage9c-seed{seed}")
    train_stage9c(
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        comp_queue=comp_q,
        attr_queue=attr_q,
        curr_queue=curr_q,
        **kwargs,
    )


def train_stage9c_parallel(
    outdir: str,
    seeds: str = "1,2,3,4,5",
    n_workers: int = 16,
    balance_archetypes: bool = False,
    cooccurrence_damp: float = 0.7,
    max_depth: int = 3,
    invalid_prob: float = 0.0,
    max_steps: int = 100_000,
    batch_size: int = 64,
    curriculum_prob: float = 0.4,
    **kwargs,
):
    """Launch Stage 9c training for all seeds in parallel."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seed_list = [int(s) for s in seeds.split(",")]

    # Shared data server
    server = SharedDataServer(
        base_seed=42,
        n_workers=n_workers,
        queue_size=4096,
        balance_archetypes=balance_archetypes,
        cooccurrence_damp=cooccurrence_damp,
        max_depth=max_depth,
        invalid_prob=invalid_prob,
    )

    # Per-seed training processes
    kwargs_training = dict(
        balance_archetypes=balance_archetypes,
        cooccurrence_damp=cooccurrence_damp,
        max_depth=max_depth,
        invalid_prob=invalid_prob,
        max_steps=max_steps,
        batch_size=batch_size,
        curriculum_prob=curriculum_prob,
        n_workers=0,  # Don't spawn additional workers; use server
        **kwargs,
    )

    procs = []
    for seed in seed_list:
        p = mp.Process(
            target=_seed_worker,
            args=(seed, str(outdir), kwargs_training, server.comp_q, server.attr_q, server.curr_q),
            name=f"stage9c-seed{seed}",
        )
        p.start()
        print(f"[seed {seed}] started PID {p.pid}")
        procs.append((seed, p))

    # Wait for all to complete
    for seed, p in procs:
        p.join()
        print(f"[seed {seed}] finished (exit code {p.exitcode})")

    server.stop()
    print("\nAll Stage 9c seeds complete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True,
                        help="Output directory for checkpoints")
    parser.add_argument("--seeds", default="1,2,3,4,5",
                        help="Comma-separated seed list")
    parser.add_argument("--max-steps", type=int, default=100_000,
                        help="Total steps across all 4 phases")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--curriculum-prob", type=float, default=0.4)
    parser.add_argument("--balance-archetypes", action="store_true")
    parser.add_argument("--n-workers", type=int, default=16,
                        help="Data server workers")
    args = parser.parse_args()

    train_stage9c_parallel(
        outdir=args.outdir,
        seeds=args.seeds,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        curriculum_prob=args.curriculum_prob,
        balance_archetypes=args.balance_archetypes,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
