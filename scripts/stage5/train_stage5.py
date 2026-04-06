"""
Stage 5: Train from scratch with adaptive freeze curriculum baked in.

Unlike Stage 4 (post-hoc fine-tuning), Stage 5 trains the encoder and decoder
jointly from step 1 under a mixed objective:
  - Normal composition + attribution batches (same as Stage 2)
  - Partial-freeze curriculum batches (each span frozen independently at 50%)
  - Full-freeze curriculum batches (all of A+BIND+B frozen)

Training is adaptive — two phases, each running until curr loss plateaus:
  Phase 1 (partial freeze): run until curr loss slope flattens over a rolling
      window. Saves a transition checkpoint at the boundary.
  Phase 2 (full freeze): run until curr loss plateaus again.

Plateau detection: compute the linear slope of curr loss over the last
`plateau_window` log intervals. If |slope| < `plateau_threshold`, the phase
is considered converged.

Transition checkpoints are saved at phase boundaries and are the primary
research artifact — they capture the state of the model at the moment each
capability stabilizes.

Usage:
    python -m whiteroom.train_stage5 \\
        --checkpoint-dir _agent/cache/runs/stage5/stage5-seed1 \\
        --seed 1 \\
        --balance-archetypes \\
        --cooccurrence-damp 0.7
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from whiteroom.generator import (
    sample_example, sample_attribution_example,
    VOCAB_SIZE, balanced_archetype_weights,
)
from whiteroom.composition import compose, find_valid_bindings
from whiteroom.model import WhiteroomTransformer
from whiteroom.vocab import Token, TRAINING_FLAGS, flag_token, port_idx_token
from whiteroom.train import collate, collate_attribution, compute_loss, DataPrefetcher
from whiteroom.finetune_curriculum import (
    _sample_curriculum_example, collate_curriculum, CurriculumPrefetcher,
)


# ---------------------------------------------------------------------------
# Bidirectional BIND helpers
# ---------------------------------------------------------------------------

def _flip_comp_example(ex, rng):
    """Randomly swap A and B in a composition Example (50% probability)."""
    if rng.random() >= 0.5:
        return ex
    a_start, a_end = ex.a_token_span
    b_start, b_end = ex.b_token_span
    a_toks  = ex.input_tokens[a_start:a_end]
    bind_toks = ex.input_tokens[a_end:b_start]  # [BIND, rel_a, rel_b]
    b_toks  = ex.input_tokens[b_start:b_end]
    tail    = ex.input_tokens[b_end:]            # any trailing tokens
    flipped = b_toks + [bind_toks[0], bind_toks[2], bind_toks[1]] + a_toks + list(tail)
    new_a_end   = len(b_toks)
    new_b_start = new_a_end + 3
    new_b_end   = new_b_start + len(a_toks)
    from dataclasses import replace as dc_replace
    return dc_replace(ex,
        input_tokens=flipped,
        a_token_span=(0, new_a_end),
        b_token_span=(new_b_start, new_b_end),
    )


def _flip_curr_sample(s, rng):
    """Randomly swap A and B in a curriculum sample tuple (50% probability)."""
    if rng.random() >= 0.5:
        return s
    input_tokens, target_tokens, a_end, b_start, b_end, extra_tok = s
    a_toks  = input_tokens[:a_end]
    bind_tok = input_tokens[a_end]
    rel_a    = input_tokens[a_end + 1]
    rel_b    = input_tokens[a_end + 2]
    b_toks  = input_tokens[b_start:b_end]
    tail    = input_tokens[b_end:]
    flipped  = b_toks + [bind_tok, rel_b, rel_a] + a_toks + list(tail)
    new_a_end   = len(b_toks)
    new_b_start = new_a_end + 3
    new_b_end   = new_b_start + len(a_toks)
    return (flipped, target_tokens, new_a_end, new_b_start, new_b_end, extra_tok)


# ---------------------------------------------------------------------------
# Shared multiprocessing data server (used by parallel launcher)
# ---------------------------------------------------------------------------

def _mp_comp_worker(seed, weights, cooccurrence_damp, max_depth,
                    invalid_prob, comp_q, attr_q, stop_event):
    import random
    rng = random.Random(seed)
    while not stop_event.is_set():
        try:
            ex = sample_example(rng, invalid_prob=invalid_prob,
                                max_depth=max_depth, weights=weights,
                                cooccurrence_damp=cooccurrence_damp)
            comp_q.put(("comp", ex), timeout=0.1)
            attr_ex = sample_attribution_example(rng, max_depth=max_depth,
                                                  weights=weights)
            attr_q.put(("attr", attr_ex), timeout=0.1)
        except Exception:
            continue


def _mp_curr_worker(seed, weights, cooccurrence_damp, curr_q, stop_event):
    import random
    rng = random.Random(seed)
    while not stop_event.is_set():
        try:
            s = _sample_curriculum_example(rng, weights=weights,
                                           cooccurrence_damp=cooccurrence_damp)
            if s is not None:
                curr_q.put(s, timeout=0.1)
        except Exception:
            continue


class SharedDataServer:
    """
    Shared multiprocessing data server for parallel Stage 5 training.
    N worker processes fill three queues (comp, attr, curr) that can be
    consumed by multiple training processes simultaneously.
    """
    def __init__(self, base_seed, n_workers=16, queue_size=2048,
                 balance_archetypes=False, cooccurrence_damp=0.0,
                 max_depth=2, invalid_prob=0.2):
        self.comp_q = mp.Queue(maxsize=queue_size)
        self.attr_q = mp.Queue(maxsize=queue_size)
        self.curr_q = mp.Queue(maxsize=queue_size)
        self._stop = mp.Event()
        self._workers = []

        weights = balanced_archetype_weights() if balance_archetypes else None
        n_comp = n_workers * 2 // 3
        n_curr = n_workers - n_comp

        for i in range(n_comp):
            p = mp.Process(
                target=_mp_comp_worker,
                args=(base_seed * 10000 + i, weights, cooccurrence_damp,
                      max_depth, invalid_prob, self.comp_q, self.attr_q,
                      self._stop),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

        for i in range(n_curr):
            p = mp.Process(
                target=_mp_curr_worker,
                args=(base_seed * 10000 + 9000 + i, weights,
                      cooccurrence_damp, self.curr_q, self._stop),
                daemon=True,
            )
            p.start()
            self._workers.append(p)

    def stop(self):
        self._stop.set()
        for p in self._workers:
            p.join(timeout=2.0)
            if p.is_alive():
                p.kill()


# ---------------------------------------------------------------------------
# Plateau detection
# ---------------------------------------------------------------------------

def _slope(values: List[float]) -> float:
    """Linear regression slope over a list of values."""
    n = len(values)
    if n < 2:
        return float("inf")
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, values))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_stage5(
    checkpoint_dir: str = "_agent/cache/runs/stage5",
    seed: int = 42,
    # Model
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    # Optimiser
    lr: float = 3e-4,
    batch_size: int = 64,
    max_steps: int = 200_000,
    # Curriculum
    curriculum_prob: float = 0.4,
    # Data — ignored if external queues provided
    balance_archetypes: bool = False,
    cooccurrence_damp: float = 0.0,
    max_depth: int = 2,
    invalid_prob: float = 0.2,
    n_workers: int = 4,
    # External shared queues (from SharedDataServer) — overrides internal prefetchers
    comp_queue=None,
    attr_queue=None,
    curr_queue=None,
    # Variants
    bidir_bind: bool = False,
    causal_encoder: bool = False,
    block_diag_encoder_mask: bool = False,
    # Logging
    log_every: int = 500,
    checkpoint_every: int = 10_000,
    # Plateau detection
    plateau_window: int = 10,       # log intervals to measure slope over
    plateau_threshold: float = 5e-5, # |slope| below this = converged
    min_phase_steps: int = 10_000,   # minimum steps before plateau can trigger
):
    import random
    rng = random.Random(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage 5 — from scratch, adaptive freeze curriculum")
    print(f"Seed: {seed}  curriculum_prob: {curriculum_prob}"
          f"  bidir_bind: {bidir_bind}  causal_encoder: {causal_encoder}")
    print(f"Plateau window: {plateau_window} intervals  threshold: {plateau_threshold}")

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"

    arch_weights = balanced_archetype_weights() if balance_archetypes else None

    # --- Model ---
    model = WhiteroomTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        causal_encoder=causal_encoder,
        block_diag_encoder_mask=block_diag_encoder_mask,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # No fixed schedule — we run until plateau, so use constant LR with
    # optional decay handled per-phase below
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
            items = [_flip_comp_example(ex, rng) for ex in items]
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
        # attribution examples don't have BIND structure — no flip needed
        return items

    def _get_curr(n):
        if use_external_queues:
            items = []
            while len(items) < n:
                items.append(curr_queue.get())
        else:
            items = curr_prefetcher.get(n)
        if bidir_bind:
            items = [_flip_curr_sample(s, rng) for s in items]
        return items

    # --- Phase tracking ---
    # phase 1 = partial freeze, phase 2 = full freeze
    phase = 1
    phase_step = 0          # steps within current phase
    curr_loss_history = []  # rolling window of per-interval avg curr loss
    phase1_transition_step = None

    running = {"seq": 0.0, "valid": 0.0, "attr": 0.0, "curr": 0.0, "total": 0.0}
    t0 = time.time()

    def save_checkpoint(path, step, note=""):
        torch.save({
            "step": step,
            "phase": phase,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": {
                "vocab_size": VOCAB_SIZE, "d_model": d_model, "nhead": nhead,
                "num_encoder_layers": num_encoder_layers,
                "num_decoder_layers": num_decoder_layers,
                "dim_feedforward": dim_feedforward, "dropout": dropout,
                "causal_encoder": causal_encoder,
                "block_diag_encoder_mask": block_diag_encoder_mask,
            },
            "note": note,
        }, path)

    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        components = {k: 0.0 for k in running}

        if rng.random() < curriculum_prob:
            # Curriculum batch — partial freeze in phase 1, full freeze in phase 2
            samples = _get_curr(batch_size)
            partial = (phase == 1)
            hybrid_mem, tgt_in, tgt_out = collate_curriculum(
                samples, device, model, rng=rng, partial_freeze=partial)

            tgt_len = tgt_in.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt_len, device=device)
            dec_out = model.decode(
                tgt_in, hybrid_mem,
                tgt_mask=tgt_mask,
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
                    "step": step, "phase": phase,
                    **avg, "lr": lr_now, "elapsed": elapsed,
                }) + "\n")

            # Track curr loss for plateau detection
            if avg["curr"] > 0:
                curr_loss_history.append(avg["curr"])
                if len(curr_loss_history) > plateau_window:
                    curr_loss_history.pop(0)

            running = {k: 0.0 for k in running}

            # --- Plateau detection & phase transition ---
            if (phase == 1
                    and phase_step >= min_phase_steps
                    and len(curr_loss_history) == plateau_window):
                slope = _slope(curr_loss_history)
                if abs(slope) < plateau_threshold:
                    print(
                        f"\n>>> Phase 1 plateau detected at step {step} "
                        f"(slope={slope:.2e}). Transitioning to phase 2.",
                        flush=True,
                    )
                    phase1_transition_step = step
                    trans_path = ckpt_dir / f"checkpoint_phase1_transition.pt"
                    save_checkpoint(trans_path, step,
                                    note=f"phase1_plateau slope={slope:.2e}")
                    print(f"  → transition checkpoint saved: {trans_path.name}",
                          flush=True)
                    phase = 2
                    phase_step = 0
                    curr_loss_history = []

            elif (phase == 2
                    and phase_step >= min_phase_steps
                    and len(curr_loss_history) == plateau_window):
                slope = _slope(curr_loss_history)
                if abs(slope) < plateau_threshold:
                    print(
                        f"\n>>> Phase 2 plateau detected at step {step} "
                        f"(slope={slope:.2e}). Training complete.",
                        flush=True,
                    )
                    break

        # --- Periodic checkpoints ---
        if step % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_{step:06d}.pt"
            save_checkpoint(ckpt_path, step)
            print(f"  → checkpoint saved: {ckpt_path.name}", flush=True)

    # Final checkpoint
    final_path = ckpt_dir / "checkpoint_final.pt"
    save_checkpoint(final_path, step, note="final")
    if prefetcher is not None:
        prefetcher.stop()
    if curr_prefetcher is not None:
        curr_prefetcher.stop()
    print(f"\nStage 5 complete at step {step}. Final checkpoint: {final_path}")
    if phase1_transition_step:
        print(f"Phase 1 → 2 transition: step {phase1_transition_step}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir",   type=str,   default="_agent/cache/runs/stage5")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--d-model",          type=int,   default=64)
    parser.add_argument("--nhead",            type=int,   default=4)
    parser.add_argument("--enc-layers",       type=int,   default=3)
    parser.add_argument("--dec-layers",       type=int,   default=3)
    parser.add_argument("--ffn-dim",          type=int,   default=256)
    parser.add_argument("--dropout",          type=float, default=0.1)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--batch-size",       type=int,   default=64)
    parser.add_argument("--max-steps",        type=int,   default=200_000)
    parser.add_argument("--curriculum-prob",  type=float, default=0.4)
    parser.add_argument("--balance-archetypes", action="store_true")
    parser.add_argument("--cooccurrence-damp",  type=float, default=0.0)
    parser.add_argument("--n-workers",        type=int,   default=4)
    parser.add_argument("--log-every",        type=int,   default=500)
    parser.add_argument("--checkpoint-every", type=int,   default=10_000)
    parser.add_argument("--plateau-window",   type=int,   default=10)
    parser.add_argument("--plateau-threshold",type=float, default=5e-5)
    parser.add_argument("--min-phase-steps",  type=int,   default=10_000)
    parser.add_argument("--bidir-bind",       action="store_true",
                        help="Train with both A→B and B→A binding orders (50/50)")
    parser.add_argument("--causal-encoder",   action="store_true",
                        help="Use unidirectional (causal) encoder")
    parser.add_argument("--block-diag-encoder-mask", action="store_true",
                        help="A and B isolated, BIND bridges (block-diagonal masking)")
    args = parser.parse_args()

    train_stage5(
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
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
        balance_archetypes=args.balance_archetypes,
        cooccurrence_damp=args.cooccurrence_damp,
        n_workers=args.n_workers,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        plateau_window=args.plateau_window,
        plateau_threshold=args.plateau_threshold,
        min_phase_steps=args.min_phase_steps,
        bidir_bind=args.bidir_bind,
        causal_encoder=args.causal_encoder,
        block_diag_encoder_mask=args.block_diag_encoder_mask,
    )
