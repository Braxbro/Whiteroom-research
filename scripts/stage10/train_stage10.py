"""
Stage 10: Asymmetric WhiteroomTransformer (thin encoder + thick decoder)

Architecture:
- Encoder (1 layer, block-diagonal masked): isolates A and B structurally
- Decoder (5 layers, free): carries the composition load

Rationale: Stage 9 used a 3-stage (enc + adaptation + dec) with symmetric 3+3
capacity. The adaptation layer was captured by whichever side learned fastest,
preventing co-adaptation. Stage 10 drops the middleman entirely — use the base
WhiteroomTransformer with block_diag_encoder_mask, shift layers to the decoder.

Both components are free from step 1, so there's no frozen-free mismatch.
Block-diagonal masking enforces isolation at the attention level without
requiring a separate adaptation stage.

Training: Same 2-phase curriculum as Stage 5/9 (partial freeze → full freeze).

Usage:
    python -m _agent.scripts.stage10.train_stage10 \\
        --checkpoint-dir _agent/cache/runs/stage10/stage10-seed1 \\
        --seed 1 \\
        --balance-archetypes
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn as nn

from whiteroom.generator import (
    sample_example, sample_attribution_example,
    VOCAB_SIZE, balanced_archetype_weights,
)
from whiteroom.model import WhiteroomTransformer
from whiteroom.vocab import Token
from whiteroom.train import collate, collate_attribution, compute_loss, DataPrefetcher
from whiteroom.finetune_curriculum import (
    _sample_curriculum_example, collate_curriculum, CurriculumPrefetcher,
)

from _agent.scripts.stage5.train_stage5 import (
    SharedDataServer, _flip_comp_example, _flip_curr_sample, _slope,
)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_stage10(
    checkpoint_dir: str = "_agent/cache/runs/stage10",
    seed: int = 42,
    # Model: thin encoder, thick decoder
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 1,
    num_decoder_layers: int = 5,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    # Optimiser
    lr: float = 3e-4,
    batch_size: int = 64,
    max_steps: int = 200_000,
    # Loss weighting
    valid_weight: float = 1.0,
    # Curriculum
    curriculum_prob: float = 0.4,
    # Data
    balance_archetypes: bool = False,
    cooccurrence_damp: float = 0.0,
    max_depth: int = 2,
    invalid_prob: float = 0.2,
    n_workers: int = 4,
    # External queues (from SharedDataServer)
    comp_queue=None,
    attr_queue=None,
    curr_queue=None,
    # Encoder masking
    block_diag_mask: bool = True,
    # LR warmup
    warmup_steps: int = 0,
    # Logging
    log_every: int = 500,
    checkpoint_every: int = 10_000,
    # Plateau detection
    plateau_window: int = 10,
    plateau_threshold: float = 5e-5,
    min_phase_steps: int = 10_000,
):
    import random
    rng = random.Random(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage 10 — Asymmetric WhiteroomTransformer (enc={num_encoder_layers} dec={num_decoder_layers}) valid_weight={valid_weight}")
    print(f"Seed: {seed}  curriculum_prob: {curriculum_prob}")
    print(f"Plateau window: {plateau_window} intervals  threshold: {plateau_threshold}")

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"

    arch_weights = balanced_archetype_weights() if balance_archetypes else None

    # --- Model: base WhiteroomTransformer, block-diagonal encoder, asymmetric layers ---
    model = WhiteroomTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        block_diag_encoder_mask=block_diag_mask,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_steps - warmup_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])
    else:
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
        return [_flip_comp_example(ex, rng) for ex in items]

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
        return [_flip_curr_sample(s, rng) for s in items]

    # --- Phase tracking ---
    phase = 1
    phase_step = 0
    curr_loss_history = []
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
                "vocab_size": VOCAB_SIZE,
                "d_model": d_model,
                "nhead": nhead,
                "num_encoder_layers": num_encoder_layers,
                "num_decoder_layers": num_decoder_layers,
                "dim_feedforward": dim_feedforward,
                "dropout": dropout,
                "block_diag_encoder_mask": block_diag_mask,
            },
            "note": note,
        }, path)

    for step in range(1, max_steps + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        components = {k: 0.0 for k in running}

        if rng.random() < curriculum_prob:
            # Curriculum batch: collate_curriculum encodes A+B, returns hybrid memory
            samples = _get_curr(batch_size)
            partial = (phase == 1)
            hybrid_mem, tgt_in, tgt_out = collate_curriculum(
                samples, device, model, rng=rng, partial_freeze=partial)

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
                valid_weight=valid_weight,
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
                    trans_path = ckpt_dir / "checkpoint_phase1_transition.pt"
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
    print(f"\nStage 10 complete at step {step}. Final checkpoint: {final_path}")
    if phase1_transition_step:
        print(f"Phase 1 → 2 transition: step {phase1_transition_step}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir",    type=str,   default="_agent/cache/runs/stage10")
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--d-model",           type=int,   default=64)
    parser.add_argument("--nhead",             type=int,   default=4)
    parser.add_argument("--enc-layers",        type=int,   default=1)
    parser.add_argument("--dec-layers",        type=int,   default=5)
    parser.add_argument("--ffn-dim",           type=int,   default=256)
    parser.add_argument("--dropout",           type=float, default=0.1)
    parser.add_argument("--lr",               type=float, default=3e-4)
    parser.add_argument("--batch-size",        type=int,   default=64)
    parser.add_argument("--max-steps",         type=int,   default=200_000)
    parser.add_argument("--valid-weight",      type=float, default=1.0)
    parser.add_argument("--curriculum-prob",   type=float, default=0.4)
    parser.add_argument("--balance-archetypes", action="store_true")
    parser.add_argument("--cooccurrence-damp", type=float, default=0.0)
    parser.add_argument("--n-workers",         type=int,   default=4)
    parser.add_argument("--log-every",         type=int,   default=500)
    parser.add_argument("--checkpoint-every",  type=int,   default=10_000)
    parser.add_argument("--no-block-diag",      action="store_true")
    parser.add_argument("--warmup-steps",      type=int,   default=0)
    parser.add_argument("--plateau-window",    type=int,   default=10)
    parser.add_argument("--plateau-threshold", type=float, default=5e-5)
    parser.add_argument("--min-phase-steps",   type=int,   default=10_000)
    args = parser.parse_args()

    train_stage10(
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
        valid_weight=args.valid_weight,
        curriculum_prob=args.curriculum_prob,
        balance_archetypes=args.balance_archetypes,
        cooccurrence_damp=args.cooccurrence_damp,
        n_workers=args.n_workers,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        block_diag_mask=not args.no_block_diag,
        warmup_steps=args.warmup_steps,
        plateau_window=args.plateau_window,
        plateau_threshold=args.plateau_threshold,
        min_phase_steps=args.min_phase_steps,
    )
