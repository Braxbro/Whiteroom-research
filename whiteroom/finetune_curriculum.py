"""
Stage 4: Curriculum fine-tuning for frozen-context decoder tolerance.

Fine-tunes a pre-trained Stage 2 checkpoint with a mixed curriculum:
  - (1 - curriculum_prob) of batches: normal composition + attribution (as Stage 2)
  - curriculum_prob of batches: property-append examples with frozen encoder spans

Curriculum batch construction:
  1. Sample a valid compound (A, B).
  2. Pick an extra flag not in A.flags | B.flags.
  3. Encode the extended sequence [A | BIND | rel_a | rel_b | B | extra_flag] fully.
  4. Detach positions 0..b_end-1 (freeze A+BIND+B spans) from the memory tensor.
     This cuts gradient flow through frozen positions — decoder must learn to
     integrate the unfrozen extra_flag token without re-encoding A or B.
  5. Decode with teacher-forcing against the full target (A.flags | B.flags | {extra_flag}).

The detach is done in-graph: memory[:, :b_end, :].detach() replaces those positions
before the decoder forward pass. The encoder still runs (needed to get valid B positions
to attend to), but gradients don't flow back through the frozen span — only through
the extra_flag position and the decoder parameters.

Usage:
    python -m whiteroom.finetune_curriculum \\
        --finetune-from _agent/cache/runs/multiseed/stage2-seed1/checkpoint_final.pt \\
        --checkpoint-dir _agent/cache/runs/stage4-seed1 \\
        --steps 20000 \\
        --curriculum-prob 0.3 \\
        --seed 1
"""

import argparse
import json
import queue
import random
import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .generator import (
    sample_example, sample_attribution_example, sample_primitive,
    serialize_entity, serialize_compound_output,
    VOCAB_SIZE, balanced_archetype_weights,
    Example,
)
from .composition import compose, find_valid_bindings
from .model import WhiteroomTransformer
from .vocab import Token, Flag, TRAINING_FLAGS, flag_token, port_idx_token
from .train import collate, collate_attribution, compute_loss, DataPrefetcher


class CurriculumPrefetcher:
    """Background workers generating curriculum examples (pure Python part only)."""

    def __init__(self, base_seed, n_workers=4, queue_size=256,
                 weights=None, cooccurrence_damp=0.0):
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._workers = []
        for i in range(n_workers):
            t = threading.Thread(
                target=self._worker,
                args=(base_seed * 1000 + i + 99999, weights, cooccurrence_damp),
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def _worker(self, seed, weights, cooccurrence_damp):
        rng = random.Random(seed)
        while not self._stop.is_set():
            try:
                s = _sample_curriculum_example(rng, weights=weights,
                                               cooccurrence_damp=cooccurrence_damp)
                if s is not None:
                    self._q.put(s, timeout=0.1)
            except queue.Full:
                continue

    def get(self, n):
        samples = []
        while len(samples) < n:
            try:
                samples.append(self._q.get(timeout=0.5))
            except queue.Empty:
                continue
        return samples

    def stop(self):
        self._stop.set()
        for t in self._workers:
            t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Curriculum example construction
# ---------------------------------------------------------------------------

def _sample_curriculum_example(
    rng: random.Random,
    weights: Optional[List[float]] = None,
    cooccurrence_damp: float = 0.0,
) -> Optional[Tuple[List[int], List[int], int, int]]:
    """
    Sample one curriculum example: (input_tokens, target_tokens, b_end, extra_flag_tok).

    input_tokens: [A | BIND | rel_a | rel_b | B | extra_flag]
    target_tokens: serialize_compound_output(compose(A,B)) with extra_flag appended
                   before END
    b_end: index in input_tokens where B's span ends (extra_flag is at b_end)

    Returns None if no valid pair found.
    """
    for _ in range(50):
        a = sample_primitive(rng, weights=weights)
        b = sample_primitive(rng, weights=weights)
        bindings = find_valid_bindings(a, b)
        if not bindings:
            continue

        # Optional co-occurrence dampening (same logic as generator)
        if cooccurrence_damp > 0.0:
            from .generator import _flags_cooccur_across
            if _flags_cooccur_across(a, b) and rng.random() < cooccurrence_damp:
                continue

        port_a_idx, port_b_idx = rng.choice(bindings)

        combined_flags = a.flags | b.flags
        available = [f for f in TRAINING_FLAGS if f not in combined_flags]
        if not available:
            continue
        extra_flag = rng.choice(available)
        extra_tok = flag_token(extra_flag)

        a_tokens, a_map = serialize_entity(a)
        b_tokens, b_map = serialize_entity(b)
        rel_a = a_map[port_a_idx]
        rel_b = b_map[port_b_idx]

        a_end   = len(a_tokens)
        b_start = a_end + 3  # BIND + rel_a + rel_b
        b_end   = b_start + len(b_tokens)

        input_tokens = (
            a_tokens
            + [Token.BIND, port_idx_token(rel_a), port_idx_token(rel_b)]
            + b_tokens
            + [extra_tok]
        )

        # Target: compound flags including extra_flag, inserted before END
        compound = compose(a, b, port_a_idx, port_b_idx)
        base_target = serialize_compound_output(compound)  # ends with END
        # Insert extra_flag before END (flags are at the tail of the output)
        assert base_target[-1] == Token.END
        target_tokens = base_target[:-1] + [extra_tok, Token.END]

        return input_tokens, target_tokens, a_end, b_start, b_end, extra_tok

    return None


def collate_curriculum(
    samples: List[Tuple[List[int], List[int], int, int, int, int]],
    device: torch.device,
    model: WhiteroomTransformer,
    rng: random.Random = None,
    partial_freeze: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build curriculum batch. Encodes each example, detaches frozen spans,
    concatenates into a padded memory tensor for batched decoding.

    partial_freeze=False (default): freeze all of A+BIND+B, extra_flag live.
    partial_freeze=True: per example, randomly choose which of A/BIND/B to
        freeze (all 8 combinations equally likely, including none frozen).
        Extra_flag position is always live.

    Returns:
        hybrid_mem:  (batch, max_src_len, d_model)  — mixed frozen/fresh
        tgt_in_pad:  (batch, max_tgt_len)
        tgt_out_pad: (batch, max_tgt_len)
    """
    mem_list, tgt_in_list, tgt_out_list = [], [], []

    for input_tokens, target_tokens, a_end, b_start, b_end, _extra_tok in samples:
        src = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            mem_full = model.encode(src)  # (1, L+1, d_model)

        if partial_freeze and rng is not None:
            # Randomly freeze each span independently
            freeze_a    = rng.random() < 0.5
            freeze_bind = rng.random() < 0.5
            freeze_b    = rng.random() < 0.5
            mask = torch.ones(mem_full.size(1), dtype=torch.bool, device=device)
            if not freeze_a:
                mask[:a_end] = False
            if not freeze_bind:
                mask[a_end:b_start] = False
            if not freeze_b:
                mask[b_start:b_end] = False
            mask[b_end:] = False  # extra_flag always live
            mem_frozen = torch.where(
                mask.unsqueeze(0).unsqueeze(-1),
                mem_full.detach(),
                mem_full,
            )
        else:
            # Freeze all of A+BIND+B, keep extra_flag live
            mem_frozen = torch.cat([
                mem_full[:, :b_end, :].detach(),
                mem_full[:, b_end:, :],
            ], dim=1)

        mem_list.append(mem_frozen.squeeze(0))  # (L+1, d_model)

        tgt = torch.tensor(target_tokens, dtype=torch.long)
        bos = torch.tensor([Token.COMPOUND], dtype=torch.long)
        tgt_in_list.append(torch.cat([bos, tgt[:-1]]))
        tgt_out_list.append(tgt)

    # Pad memory sequences
    max_src = max(m.size(0) for m in mem_list)
    d = mem_list[0].size(1)
    hybrid_mem = torch.zeros(len(mem_list), max_src, d, device=device)
    for i, m in enumerate(mem_list):
        hybrid_mem[i, :m.size(0), :] = m

    tgt_in_pad  = pad_sequence(tgt_in_list,  batch_first=True, padding_value=Token.PAD).to(device)
    tgt_out_pad = pad_sequence(tgt_out_list, batch_first=True, padding_value=Token.PAD).to(device)

    return hybrid_mem, tgt_in_pad, tgt_out_pad


# ---------------------------------------------------------------------------
# Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(
    finetune_from: str,
    steps: int = 20_000,
    batch_size: int = 64,
    lr: float = 1e-4,
    checkpoint_dir: str = "_agent/cache/runs/stage4",
    log_every: int = 500,
    checkpoint_every: int = 5000,
    seed: int = 42,
    curriculum_prob: float = 0.3,
    balance_archetypes: bool = False,
    cooccurrence_damp: float = 0.0,
    partial_freeze: bool = False,
    freeze_encoder: bool = False,
    n_workers: int = 4,
):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Fine-tuning from: {finetune_from}")
    print(f"Curriculum prob: {curriculum_prob}  partial_freeze: {partial_freeze}")

    arch_weights = balanced_archetype_weights() if balance_archetypes else None
    prefetcher = DataPrefetcher(
        base_seed=seed, n_workers=n_workers,
        balance_archetypes=balance_archetypes,
        cooccurrence_damp=cooccurrence_damp,
        attribution=True,
    )
    curr_prefetcher = CurriculumPrefetcher(
        base_seed=seed, n_workers=n_workers,
        weights=arch_weights,
        cooccurrence_damp=cooccurrence_damp,
    )
    print(f"Data prefetchers: {n_workers} workers each")

    # Load checkpoint
    ckpt = torch.load(finetune_from, map_location=device, weights_only=False)
    model = WhiteroomTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}  (loaded from step {ckpt.get('step', '?')})")

    if freeze_encoder:
        for p in model.src_embed.parameters():
            p.requires_grad_(False)
        for p in model.transformer.encoder.parameters():
            p.requires_grad_(False)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Encoder frozen. Trainable params: {n_trainable:,}")

    # Lower LR for fine-tuning
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    seq_loss_fn   = nn.CrossEntropyLoss(ignore_index=Token.PAD)
    valid_loss_fn = nn.BCEWithLogitsLoss()

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"

    running = {"seq": 0.0, "valid": 0.0, "attr": 0.0, "curr": 0.0, "total": 0.0}
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        components = {k: 0.0 for k in running}

        if rng.random() < curriculum_prob:
            # --- Curriculum batch: frozen-span property-append ---
            curr_size = batch_size
            samples = curr_prefetcher.get(curr_size)

            hybrid_mem, tgt_in, tgt_out = collate_curriculum(
                samples, device, model, rng=rng, partial_freeze=partial_freeze)

            tgt_len = tgt_in.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
            tgt_pad_mask = (tgt_in == Token.PAD)

            dec_out = model.decode(
                tgt_in, hybrid_mem,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
            )
            logits = model.seq_head(dec_out)  # (batch, tgt_len, vocab)

            b, t, v = logits.shape
            curr_loss = seq_loss_fn(logits.reshape(b * t, v), tgt_out.reshape(b * t))
            total_loss = total_loss + curr_loss
            components["curr"] = curr_loss.item()
            components["total"] = total_loss.item()

        else:
            # --- Normal Stage 2 batch: composition + attribution ---
            comp_size = batch_size // 2
            comp_examples = prefetcher.get_comp(comp_size)
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

            attr_size = batch_size - comp_size
            attr_examples = prefetcher.get_attr(attr_size)
            attr_batch = collate_attribution(attr_examples, device)
            attr_logits, _ = model(
                attr_batch["src"], attr_batch["tgt_in"],
                src_key_padding_mask=attr_batch["src_pad_mask"],
                tgt_key_padding_mask=attr_batch["tgt_pad_mask"],
            )
            b, t, v = attr_logits.shape
            attr_loss = seq_loss_fn(attr_logits.reshape(b * t, v),
                                    attr_batch["tgt_out"].reshape(b * t))
            total_loss = total_loss + attr_loss
            components["attr"]  = attr_loss.item()
            components["total"] = total_loss.item()

        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        for k in running:
            running[k] += components[k]

        if step % log_every == 0:
            elapsed = time.time() - t0
            avg = {k: v / log_every for k, v in running.items()}
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"step {step:6d} | "
                f"loss {avg['total']:.4f} | "
                f"seq {avg['seq']:.4f} | "
                f"attr {avg['attr']:.4f} | "
                f"curr {avg['curr']:.4f} | "
                f"lr {lr_now:.2e} | "
                f"{elapsed:.0f}s",
                flush=True,
            )
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, **avg, "lr": lr_now, "elapsed": elapsed}) + "\n")
            running = {k: 0.0 for k in running}

        if step % checkpoint_every == 0:
            ckpt_path = ckpt_dir / f"checkpoint_{step:06d}.pt"
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": ckpt["config"],
                "finetune_from": finetune_from,
                "curriculum_prob": curriculum_prob,
            }, ckpt_path)
            print(f"  → checkpoint saved: {ckpt_path.name}", flush=True)

    final_path = ckpt_dir / "checkpoint_final.pt"
    torch.save({
        "step": steps,
        "model_state": model.state_dict(),
        "config": ckpt["config"],
        "finetune_from": finetune_from,
        "curriculum_prob": curriculum_prob,
        "partial_freeze": partial_freeze,
    }, final_path)
    prefetcher.stop()
    curr_prefetcher.stop()
    print(f"Fine-tuning complete. Final checkpoint: {final_path}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune-from", required=True,
                        help="Path to Stage 2 checkpoint to fine-tune from")
    parser.add_argument("--steps",            type=int,   default=20_000)
    parser.add_argument("--batch-size",       type=int,   default=64)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--checkpoint-dir",   type=str,   default="_agent/cache/runs/stage4")
    parser.add_argument("--log-every",        type=int,   default=500)
    parser.add_argument("--checkpoint-every", type=int,   default=5000)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--curriculum-prob",  type=float, default=0.3,
                        help="Fraction of steps using frozen-span curriculum batches")
    parser.add_argument("--balance-archetypes", action="store_true")
    parser.add_argument("--cooccurrence-damp",  type=float, default=0.0)
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Freeze src_embed + transformer.encoder weights entirely; "
                             "only decoder and heads are trained.")
    parser.add_argument("--partial-freeze", action="store_true",
                        help="Randomly vary which spans are frozen per curriculum batch "
                             "(A, BIND, B each independently 50%% frozen). "
                             "Default: freeze all of A+BIND+B.")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Number of background data generation workers")
    args = parser.parse_args()

    finetune(
        finetune_from=args.finetune_from,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        curriculum_prob=args.curriculum_prob,
        balance_archetypes=args.balance_archetypes,
        cooccurrence_damp=args.cooccurrence_damp,
        partial_freeze=args.partial_freeze,
        freeze_encoder=args.freeze_encoder,
        n_workers=args.n_workers,
    )
