"""
Stage 5 parallel launcher: trains all seeds simultaneously with a shared
data generation pool.

One SharedDataServer fills three multiprocessing queues (comp, attr, curr)
using N_WORKERS CPU processes. All seed training processes pull from the
same queues, so CPU data generation is never duplicated.

Each seed runs train_stage5() in its own process with its own model,
optimizer, and RNG state, but shares the data stream.

Usage:
    python -m whiteroom.train_stage5_parallel \\
        --outdir _agent/cache/runs/stage5 \\
        --seeds 1,2,3,4,5 \\
        --n-workers 16 \\
        --balance-archetypes \\
        --cooccurrence-damp 0.7
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

from _agent.scripts.stage5.train_stage5 import train_stage5, SharedDataServer


def _seed_worker(seed, outdir, kwargs, comp_q, attr_q, curr_q):
    """Entry point for each per-seed training process."""
    # Each process needs its own CUDA context
    import torch
    torch.manual_seed(seed)

    checkpoint_dir = str(Path(outdir) / f"stage5-seed{seed}")
    train_stage5(
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        comp_queue=comp_q,
        attr_queue=attr_q,
        curr_queue=curr_q,
        **kwargs,
    )


def launch_parallel(
    outdir: str = "_agent/cache/runs/stage5",
    seeds: list = None,
    n_workers: int = 16,
    balance_archetypes: bool = False,
    cooccurrence_damp: float = 0.0,
    # Passed through to train_stage5
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    lr: float = 3e-4,
    batch_size: int = 64,
    max_steps: int = 200_000,
    curriculum_prob: float = 0.4,
    log_every: int = 500,
    checkpoint_every: int = 10_000,
    plateau_window: int = 10,
    plateau_threshold: float = 5e-5,
    min_phase_steps: int = 10_000,
    bidir_bind: bool = False,
    causal_encoder: bool = False,
    block_diag_encoder_mask: bool = False,
):
    if seeds is None:
        seeds = [1, 2, 3, 4, 5]

    Path(outdir).mkdir(parents=True, exist_ok=True)

    print(f"Launching {len(seeds)} seeds in parallel with {n_workers} shared data workers")
    print(f"Seeds: {seeds}  outdir: {outdir}")

    # Shared data server
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
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        max_steps=max_steps,
        curriculum_prob=curriculum_prob,
        log_every=log_every,
        checkpoint_every=checkpoint_every,
        plateau_window=plateau_window,
        plateau_threshold=plateau_threshold,
        min_phase_steps=min_phase_steps,
        balance_archetypes=balance_archetypes,
        cooccurrence_damp=cooccurrence_damp,
        bidir_bind=bidir_bind,
        causal_encoder=causal_encoder,
        block_diag_encoder_mask=block_diag_encoder_mask,
    )

    # Spawn one process per seed
    procs = []
    for seed in seeds:
        ckpt_dir = Path(outdir) / f"stage5-seed{seed}"
        if (ckpt_dir / "checkpoint_final.pt").exists():
            print(f"[seed {seed}] already done, skipping")
            continue

        p = mp.Process(
            target=_seed_worker,
            args=(seed, outdir, train_kwargs,
                  server.comp_q, server.attr_q, server.curr_q),
            name=f"stage5-seed{seed}",
        )
        p.start()
        print(f"[seed {seed}] started PID {p.pid}")
        procs.append((seed, p))

    # Wait for all to finish
    for seed, p in procs:
        p.join()
        print(f"[seed {seed}] finished (exit code {p.exitcode})")

    server.stop()
    print("\nAll seeds complete.")


if __name__ == "__main__":
    # Required for multiprocessing on Linux with spawn start method
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir",            type=str,   default="_agent/cache/runs/stage5")
    parser.add_argument("--seeds",             type=str,   default="1,2,3,4,5")
    parser.add_argument("--n-workers",         type=int,   default=16)
    parser.add_argument("--balance-archetypes", action="store_true")
    parser.add_argument("--cooccurrence-damp", type=float, default=0.0)
    parser.add_argument("--d-model",           type=int,   default=64)
    parser.add_argument("--nhead",             type=int,   default=4)
    parser.add_argument("--enc-layers",        type=int,   default=3)
    parser.add_argument("--dec-layers",        type=int,   default=3)
    parser.add_argument("--ffn-dim",           type=int,   default=256)
    parser.add_argument("--dropout",           type=float, default=0.1)
    parser.add_argument("--lr",                type=float, default=3e-4)
    parser.add_argument("--batch-size",        type=int,   default=64)
    parser.add_argument("--max-steps",         type=int,   default=200_000)
    parser.add_argument("--curriculum-prob",   type=float, default=0.4)
    parser.add_argument("--log-every",         type=int,   default=500)
    parser.add_argument("--checkpoint-every",  type=int,   default=10_000)
    parser.add_argument("--plateau-window",    type=int,   default=10)
    parser.add_argument("--plateau-threshold", type=float, default=5e-5)
    parser.add_argument("--min-phase-steps",   type=int,   default=10_000)
    parser.add_argument("--bidir-bind",        action="store_true")
    parser.add_argument("--causal-encoder",    action="store_true")
    parser.add_argument("--block-diag-encoder-mask",  action="store_true")
    args = parser.parse_args()

    launch_parallel(
        outdir=args.outdir,
        seeds=[int(s) for s in args.seeds.split(",")],
        n_workers=args.n_workers,
        balance_archetypes=args.balance_archetypes,
        cooccurrence_damp=args.cooccurrence_damp,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ffn_dim,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        curriculum_prob=args.curriculum_prob,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        plateau_window=args.plateau_window,
        plateau_threshold=args.plateau_threshold,
        min_phase_steps=args.min_phase_steps,
        bidir_bind=args.bidir_bind,
        causal_encoder=args.causal_encoder,
        block_diag_encoder_mask=args.block_diag_encoder_mask,
    )
