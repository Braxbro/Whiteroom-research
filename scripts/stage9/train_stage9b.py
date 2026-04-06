"""
Stage 9b: 4-Phase Curriculum with Selective Freezing

Hypothesis: Stage 9's encoder is so strong from block-diagonal training that it
"bullies" the bridge into following along, preventing the decoder from learning.

Solution: Run the standard 2-phase curriculum TWICE:
  - Pass 1 (Phases 1A & 2A): Bridge FROZEN, encoder/decoder free
    Allows encoder and decoder to find compatibility without bridge moving.

  - Pass 2 (Phases 1B & 2B): Encoder FROZEN, bridge free
    Allows bridge to adapt to encoder output and decoder to work with bridge.

This prevents the encoder from modifying the representation space while the
bridge is learning to adapt it.

Usage:
    python -m _agent.scripts.stage9.train_stage9b \\
        --checkpoint-dir _agent/cache/runs/stage9b/stage9b-seed1 \\
        --seed 1
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from whiteroom.generator import (
    sample_example, sample_attribution_example,
    VOCAB_SIZE, balanced_archetype_weights,
)
from whiteroom.composition import compose, find_valid_bindings
from whiteroom.model import WhiteroomTransformer3Stage
from whiteroom.vocab import Token, TRAINING_FLAGS, flag_token, port_idx_token
from whiteroom.train import collate, collate_attribution, compute_loss, DataPrefetcher
from whiteroom.finetune_curriculum import (
    _sample_curriculum_example, collate_curriculum, CurriculumPrefetcher,
)

# SharedDataServer is imported from stage5
from _agent.scripts.stage5.train_stage5 import (
    SharedDataServer, _flip_comp_example, _flip_curr_sample, _slope, _mp_comp_worker, _mp_curr_worker
)


def train_stage9b(
    checkpoint_dir: str,
    seed: int = 1,
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    max_seq_len: int = 256,
    lr: float = 3e-4,
    batch_size: int = 64,
    max_steps: int = 100_000,  # Total for all 4 phases
    curriculum_prob: float = 0.4,
    bidir_bind: bool = False,
    balance_archetypes: bool = False,
    cooccurrence_damp: float = 0.0,
    max_depth: int = 3,
    invalid_prob: float = 0.0,
    plateau_window: int = 10,
    plateau_threshold: float = 5e-5,
    min_phase_steps: int = 10_000,
    log_every: int = 500,
    checkpoint_every: int = 10_000,
    n_workers: int = 4,
    comp_queue: Optional[mp.Queue] = None,
    attr_queue: Optional[mp.Queue] = None,
    curr_queue: Optional[mp.Queue] = None,
):
    """Train Stage 9b with 4-phase curriculum."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    rng_cpu = __import__("random").Random(seed)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"

    arch_weights = balanced_archetype_weights() if balance_archetypes else None

    # --- Model: 3-Stage ---
    model = WhiteroomTransformer3Stage(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps)

    seq_loss_fn   = nn.CrossEntropyLoss(ignore_index=Token.PAD)
    valid_loss_fn = nn.BCEWithLogitsLoss()

    # --- Prefetchers or external queues ---
    use_external_queues = (comp_queue is not None)
    if use_external_queues:
        print("Using shared external data queues")
        prefetcher = None
        curr_prefetcher = None
    else:
        prefetcher = DataPrefetcher(
            base_seed=seed, n_workers=n_workers,
            balance_archetypes=balance_archetypes,
            cooccurrence_damp=cooccurrence_damp,
            max_depth=max_depth,
            invalid_prob=invalid_prob,
            attribution=True,
        )
        curr_prefetcher = CurriculumPrefetcher(
            base_seed=seed, n_workers=n_workers,
            weights=arch_weights,
            cooccurrence_damp=cooccurrence_damp,
        )
        print(f"Data prefetchers: {n_workers} workers each")

    def _get_comp(n):
        if use_external_queues:
            items = []
            while len(items) < n:
                kind, ex = comp_queue.get()
                if kind == "comp":
                    items.append(ex)
                else:
                    attr_queue.put((kind, ex))
        else:
            items = prefetcher.get_comp(n)
        if bidir_bind:
            items = [_flip_comp_example(ex, rng_cpu) for ex in items]
        return items

    def _get_attr(n):
        if use_external_queues:
            items = []
            while len(items) < n:
                kind, ex = attr_queue.get()
                if kind == "attr":
                    items.append(ex)
                else:
                    comp_queue.put((kind, ex))
        else:
            items = prefetcher.get_attr(n)
        return items

    def _get_curr(n):
        if use_external_queues:
            items = []
            while len(items) < n:
                items.append(curr_queue.get())
        else:
            items = curr_prefetcher.get(n)
        if bidir_bind:
            items = [_flip_curr_sample(s, rng_cpu) for s in items]
        return items

    # --- Phase tracking ---
    # Phases: 1A, 2A (bridge frozen), 1B, 2B (encoder frozen)
    phase_sequence = ["1A", "2A", "1B", "2B"]
    phase_idx = 0
    phase = phase_sequence[phase_idx]
    phase_step = 0
    curr_loss_history = []
    phase_transition_steps = {}

    running = {"seq": 0.0, "valid": 0.0, "attr": 0.0, "curr": 0.0, "total": 0.0}
    t0 = time.time()

    def save_checkpoint(path, step, note=""):
        torch.save({
            "step": step,
            "phase": phase,
            "phase_idx": phase_idx,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": {
                "vocab_size": VOCAB_SIZE,
                "d_model": d_model,
                "nhead": nhead,
                "num_encoder_layers": num_encoder_layers,
                "num_decoder_layers": num_decoder_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
                "model_type": "3stage",
            },
            "note": note,
        }, path)

    def set_freezing(phase_str: str):
        """Apply freezing based on current phase."""
        if phase_str in ("1A", "2A"):
            # Bridge frozen, encoder/decoder free
            model.adaptation.requires_grad_(False)
            model.encoder.requires_grad_(True)
            model.decoder.requires_grad_(True)
            print(f">>> Phase {phase_str}: Bridge FROZEN, encoder/decoder FREE")
        elif phase_str in ("1B", "2B"):
            # Encoder frozen, bridge/decoder free
            model.encoder.requires_grad_(False)
            model.adaptation.requires_grad_(True)
            model.decoder.requires_grad_(True)
            print(f">>> Phase {phase_str}: Encoder FROZEN, bridge/decoder FREE")

    # Initialize freezing for phase 1A
    set_freezing(phase_sequence[0])

    # --- Training loop ---
    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        components = {k: 0.0 for k in running}

        if rng_cpu.random() < curriculum_prob:
            # Curriculum batch
            samples = _get_curr(batch_size)
            # Partial freeze only in phases 1A and 1B (not 2A/2B)
            partial = (phase in ("1A", "1B"))
            hybrid_mem, tgt_in, tgt_out = collate_curriculum(
                samples, device, model, rng=rng_cpu, partial_freeze=partial)

            # For 3-stage model: apply adaptation to hybrid_mem
            hybrid_mem = model.adapt(hybrid_mem)

            tgt_len = tgt_in.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=device)
            dec_out = model.decode(
                tgt_in, hybrid_mem,
                tgt_key_padding_mask=(tgt_in == Token.PAD),
            )
            logits = model.seq_head(dec_out)
            b, t, v = logits.shape
            curr_loss = seq_loss_fn(logits.reshape(b * t, v), tgt_out.reshape(b * t))
            total_loss = total_loss + curr_loss
            components["curr"] = curr_loss.item()

        else:
            # Normal composition + attribution batch
            comp_size = batch_size // 2
            comp_examples = _get_comp(comp_size)
            comp_batch = collate(comp_examples, device)
            seq_logits, valid_logits = model(
                comp_batch["src"], comp_batch["tgt_in"],
                src_key_padding_mask=comp_batch["src_pad_mask"],
                tgt_key_padding_mask=comp_batch["tgt_pad_mask"],
            )
            comp_loss, comp_components = compute_loss(
                seq_logits, valid_logits,
                comp_batch["tgt_out"], comp_batch["is_valid"],
                seq_loss_fn, valid_loss_fn,
            )
            total_loss = total_loss + comp_loss
            components["seq"]   = comp_components["seq"]
            components["valid"] = comp_components["valid"]

            attr_examples = _get_attr(batch_size - comp_size)
            attr_batch = collate_attribution(attr_examples, device)
            attr_logits, _ = model(
                attr_batch["src"], attr_batch["tgt_in"],
                src_key_padding_mask=attr_batch["src_pad_mask"],
                tgt_key_padding_mask=attr_batch["tgt_pad_mask"],
            )
            b, t, v = attr_logits.shape
            attr_loss = seq_loss_fn(
                attr_logits.reshape(b * t, v),
                attr_batch["tgt_out"].reshape(b * t),
            )
            total_loss = total_loss + attr_loss
            components["attr"] = attr_loss.item()

        components["total"] = total_loss.item()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        for k in running:
            running[k] += components[k]

        phase_step += 1

        # --- Logging ---
        if step % log_every == 0:
            elapsed = time.time() - t0
            avg = {k: v / log_every for k, v in running.items()}
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"step {step:6d} [ph{phase}] | "
                f"loss {avg['total']:.4f} | "
                f"seq {avg['seq']:.4f} | "
                f"curr {avg['curr']:.4f} | "
                f"lr {lr_now:.2e} | "
                f"{elapsed:.0f}s",
                flush=True,
            )
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "phase": phase, "phase_idx": phase_idx,
                    **avg, "lr": lr_now, "elapsed": elapsed,
                }) + "\n")

            # Track curr loss for plateau detection
            if avg["curr"] > 0:
                curr_loss_history.append(avg["curr"])
                if len(curr_loss_history) > plateau_window:
                    curr_loss_history.pop(0)

            running = {k: 0.0 for k in running}

            # --- Plateau detection & phase transition ---
            if (phase_step >= min_phase_steps
                    and len(curr_loss_history) == plateau_window):
                slope = _slope(curr_loss_history)
                if abs(slope) < plateau_threshold:
                    phase_transition_steps[phase] = step
                    print(
                        f"\n>>> Phase {phase} plateau detected at step {step} "
                        f"(slope={slope:.2e}).",
                        flush=True,
                    )

                    # Save transition checkpoint
                    trans_path = ckpt_dir / f"checkpoint_phase{phase}_transition.pt"
                    save_checkpoint(trans_path, step,
                                    note=f"phase{phase}_plateau slope={slope:.2e}")
                    print(f"  → transition checkpoint: {trans_path.name}",
                          flush=True)

                    # Move to next phase
                    if phase_idx < len(phase_sequence) - 1:
                        phase_idx += 1
                        phase = phase_sequence[phase_idx]
                        phase_step = 0
                        curr_loss_history = []
                        set_freezing(phase)
                    else:
                        # All phases complete
                        print(
                            f"\n>>> All 4 phases complete. Training finished.",
                            flush=True,
                        )
                        break

        # --- Periodic checkpoints ---
        if step % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_{step:06d}.pt"
            save_checkpoint(ckpt_path, step)
            print(f"  → checkpoint: {ckpt_path.name}", flush=True)

    # Final checkpoint
    final_path = ckpt_dir / "checkpoint_final.pt"
    save_checkpoint(final_path, step, note="final")
    print(f"\nFinal checkpoint: {final_path}")
    print(f"Phase transitions: {phase_transition_steps}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--curriculum-prob", type=float, default=0.4)
    parser.add_argument("--bidir-bind", action="store_true")
    parser.add_argument("--balance-archetypes", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--min-phase-steps", type=int, default=10_000)
    parser.add_argument("--plateau-threshold", type=float, default=5e-5)
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()

    train_stage9b(
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        curriculum_prob=args.curriculum_prob,
        bidir_bind=args.bidir_bind,
        balance_archetypes=args.balance_archetypes,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        min_phase_steps=args.min_phase_steps,
        plateau_threshold=args.plateau_threshold,
        n_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
