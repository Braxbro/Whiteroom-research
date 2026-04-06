"""
Training loop for the whiteroom transformer.

Data is generated on-the-fly — no static dataset.
Checkpoints and logs are written to _agent/cache/.

Usage:
    python -m whiteroom.train [--steps N] [--batch-size N] [--checkpoint-dir PATH]
"""

import argparse
import os
import queue
import random
import threading
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .generator import (
    sample_example, sample_attribution_example, Example, AttributionExample,
    VOCAB_SIZE, balanced_archetype_weights,
)
from .model import WhiteroomTransformer
from .vocab import Token


# ---------------------------------------------------------------------------
# Background data prefetcher
# ---------------------------------------------------------------------------

class DataPrefetcher:
    """
    Runs N worker threads that continuously generate examples and push them
    into a bounded queue. The training loop pops from the queue instead of
    generating inline, keeping the GPU fed while CPU sampling runs in parallel.

    Each worker gets its own Random instance seeded deterministically from
    the base seed so results are reproducible.
    """

    def __init__(
        self,
        base_seed: int,
        n_workers: int = 4,
        queue_size: int = 256,
        balance_archetypes: bool = False,
        cooccurrence_damp: float = 0.0,
        max_depth: int = 2,
        invalid_prob: float = 0.2,
        attribution: bool = True,
    ):
        self._q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._workers: List[threading.Thread] = []

        from .generator import balanced_archetype_weights
        weights = balanced_archetype_weights() if balance_archetypes else None

        for i in range(n_workers):
            worker_seed = base_seed * 1000 + i
            t = threading.Thread(
                target=self._worker,
                args=(worker_seed, weights, cooccurrence_damp, max_depth,
                      invalid_prob, attribution),
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def _worker(self, seed, weights, cooccurrence_damp, max_depth,
                invalid_prob, attribution):
        rng = random.Random(seed)
        while not self._stop.is_set():
            try:
                ex = sample_example(
                    rng, invalid_prob=invalid_prob, max_depth=max_depth,
                    weights=weights, cooccurrence_damp=cooccurrence_damp,
                )
                self._q.put(("comp", ex), timeout=0.1)
                if attribution:
                    attr_ex = sample_attribution_example(
                        rng, max_depth=max_depth, weights=weights)
                    self._q.put(("attr", attr_ex), timeout=0.1)
            except queue.Full:
                continue

    def get_comp(self, n: int) -> list:
        examples = []
        while len(examples) < n:
            kind, ex = self._q.get()
            if kind == "comp":
                examples.append(ex)
            else:
                try:
                    self._q.put((kind, ex), timeout=0.1)  # put attr examples back
                except queue.Full:
                    pass  # drop and let workers refill
        return examples

    def get_attr(self, n: int) -> list:
        examples = []
        while len(examples) < n:
            kind, ex = self._q.get()
            if kind == "attr":
                examples.append(ex)
            else:
                try:
                    self._q.put((kind, ex), timeout=0.1)  # put comp examples back
                except queue.Full:
                    pass  # drop and let workers refill
        return examples

    def stop(self):
        self._stop.set()
        for t in self._workers:
            t.join(timeout=1.0)


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate(examples: List[Example], device: torch.device) -> dict:
    """
    Pad composition examples into batched tensors.
    """
    src_list, tgt_in_list, tgt_out_list, valid_list = [], [], [], []

    for ex in examples:
        src = torch.tensor(ex.input_tokens, dtype=torch.long)
        tgt = torch.tensor(ex.target_tokens, dtype=torch.long)
        bos = torch.tensor([Token.COMPOUND], dtype=torch.long)

        src_list.append(src)
        tgt_in_list.append(torch.cat([bos, tgt[:-1]]))
        tgt_out_list.append(tgt)
        valid_list.append(float(ex.is_valid))

    src_pad = pad_sequence(src_list, batch_first=True, padding_value=Token.PAD).to(device)
    tgt_in_pad = pad_sequence(tgt_in_list, batch_first=True, padding_value=Token.PAD).to(device)
    tgt_out_pad = pad_sequence(tgt_out_list, batch_first=True, padding_value=Token.PAD).to(device)
    is_valid = torch.tensor(valid_list, dtype=torch.float, device=device).unsqueeze(1)

    return {
        "src": src_pad,
        "tgt_in": tgt_in_pad,
        "tgt_out": tgt_out_pad,
        "is_valid": is_valid,
        "src_pad_mask": (src_pad == Token.PAD),
        "tgt_pad_mask": (tgt_in_pad == Token.PAD),
    }


def collate_attribution(examples: List[AttributionExample], device: torch.device) -> dict:
    """
    Pad attribution examples into batched tensors.
    is_valid is always 1.0 (attribution only generated for valid compounds).
    """
    src_list, tgt_in_list, tgt_out_list = [], [], []

    for ex in examples:
        src = torch.tensor(ex.input_tokens, dtype=torch.long)
        tgt = torch.tensor(ex.target_tokens, dtype=torch.long)
        bos = torch.tensor([Token.COMPOUND], dtype=torch.long)

        src_list.append(src)
        tgt_in_list.append(torch.cat([bos, tgt[:-1]]))
        tgt_out_list.append(tgt)

    src_pad = pad_sequence(src_list, batch_first=True, padding_value=Token.PAD).to(device)
    tgt_in_pad = pad_sequence(tgt_in_list, batch_first=True, padding_value=Token.PAD).to(device)
    tgt_out_pad = pad_sequence(tgt_out_list, batch_first=True, padding_value=Token.PAD).to(device)
    is_valid = torch.ones(len(examples), 1, dtype=torch.float, device=device)

    return {
        "src": src_pad,
        "tgt_in": tgt_in_pad,
        "tgt_out": tgt_out_pad,
        "is_valid": is_valid,
        "src_pad_mask": (src_pad == Token.PAD),
        "tgt_pad_mask": (tgt_in_pad == Token.PAD),
    }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    seq_logits: torch.Tensor,   # (batch, tgt_len, vocab_size)
    valid_logits: torch.Tensor, # (batch, 1)
    tgt_out: torch.Tensor,      # (batch, tgt_len)
    is_valid: torch.Tensor,     # (batch, 1)
    seq_loss_fn: nn.CrossEntropyLoss,
    valid_loss_fn: nn.BCEWithLogitsLoss,
    valid_weight: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    # Sequence loss — only on valid examples (is_valid == 1)
    # For invalid examples, target is just [END], which is short and fine to include
    batch, tgt_len, vocab = seq_logits.shape
    seq_loss = seq_loss_fn(
        seq_logits.reshape(batch * tgt_len, vocab),
        tgt_out.reshape(batch * tgt_len),
    )
    v_loss = valid_loss_fn(valid_logits, is_valid)
    total = seq_loss + valid_weight * v_loss
    return total, {"seq": seq_loss.item(), "valid": v_loss.item(), "total": total.item()}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    steps: int = 20_000,
    batch_size: int = 64,
    lr: float = 3e-4,
    checkpoint_dir: str = "/home/babrook/Documents/research/_agent/cache",
    log_every: int = 100,
    checkpoint_every: int = 2000,
    seed: int = 42,
    max_depth: int = 2,
    invalid_prob: float = 0.2,
    attribution: bool = True,
    balance_archetypes: bool = False,
    cooccurrence_damp: float = 0.0,
    n_workers: int = 4,
    # Model hyperparameters
    d_model: int = 64,
    nhead: int = 4,
    num_encoder_layers: int = 3,
    num_decoder_layers: int = 3,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Vocab size: {VOCAB_SIZE}")

    arch_weights = balanced_archetype_weights() if balance_archetypes else None
    if balance_archetypes:
        print(f"Balanced archetype weights: {arch_weights}")
    if cooccurrence_damp > 0.0:
        print(f"Co-occurrence damp: {cooccurrence_damp}")

    prefetcher = DataPrefetcher(
        base_seed=seed,
        n_workers=n_workers,
        balance_archetypes=balance_archetypes,
        cooccurrence_damp=cooccurrence_damp,
        max_depth=max_depth,
        invalid_prob=invalid_prob,
        attribution=attribution,
    )
    print(f"Data prefetcher: {n_workers} workers")

    model = WhiteroomTransformer(
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    seq_loss_fn = nn.CrossEntropyLoss(ignore_index=Token.PAD)
    valid_loss_fn = nn.BCEWithLogitsLoss()

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"

    running = {"seq": 0.0, "valid": 0.0, "attr": 0.0, "total": 0.0}
    t0 = time.time()

    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        components = {"seq": 0.0, "valid": 0.0, "attr": 0.0, "total": 0.0}

        # Composition batch (full batch when attribution disabled, half when enabled)
        comp_size = batch_size // 2 if attribution else batch_size
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
        components["seq"]   += comp_components["seq"]
        components["valid"] += comp_components["valid"]

        # Attribution batch (optional)
        if attribution:
            attr_size = batch_size - comp_size
            attr_examples = prefetcher.get_attr(attr_size)
            attr_batch = collate_attribution(attr_examples, device)
            attr_seq_logits, _ = model(
                attr_batch["src"], attr_batch["tgt_in"],
                src_key_padding_mask=attr_batch["src_pad_mask"],
                tgt_key_padding_mask=attr_batch["tgt_pad_mask"],
            )
            b, t, v = attr_seq_logits.shape
            attr_loss = seq_loss_fn(
                attr_seq_logits.reshape(b * t, v),
                attr_batch["tgt_out"].reshape(b * t),
            )
            total_loss = total_loss + attr_loss
            components["attr"] += attr_loss.item()

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
                f"valid {avg['valid']:.4f} | "
                f"attr {avg['attr']:.4f} | "
                f"lr {lr_now:.2e} | "
                f"{elapsed:.0f}s"
            )
            import json
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
                "config": {
                    "vocab_size": VOCAB_SIZE, "d_model": d_model, "nhead": nhead,
                    "num_encoder_layers": num_encoder_layers,
                    "num_decoder_layers": num_decoder_layers,
                    "dim_feedforward": dim_feedforward, "dropout": dropout,
                },
            }, ckpt_path)
            print(f"  → checkpoint saved: {ckpt_path.name}")

    # Final checkpoint
    final_path = ckpt_dir / "checkpoint_final.pt"
    torch.save({
        "step": steps,
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": VOCAB_SIZE, "d_model": d_model, "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "dim_feedforward": dim_feedforward, "dropout": dropout,
        },
    }, final_path)
    prefetcher.stop()
    print(f"Training complete. Final checkpoint: {final_path}")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/home/babrook/Documents/research/_agent/cache")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--enc-layers", type=int, default=3)
    parser.add_argument("--dec-layers", type=int, default=3)
    parser.add_argument("--ffn-dim", type=int, default=256)
    parser.add_argument("--no-attribution", action="store_true",
                        help="Disable attribution task (Stage 1 mode)")
    parser.add_argument("--balance-archetypes", action="store_true",
                        help="Reweight archetype sampling to equalise per-flag frequency "
                             "and downweight zero-flag archetypes")
    parser.add_argument("--cooccurrence-damp", type=float, default=0.0,
                        help="Probability of rejecting pairs where co-occurring flag pairs "
                             "are split across A and B (0=off, 0.7=strong dampening)")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Number of background data generation workers")
    args = parser.parse_args()

    train(
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        seed=args.seed,
        attribution=not args.no_attribution,
        balance_archetypes=args.balance_archetypes,
        cooccurrence_damp=args.cooccurrence_damp,
        n_workers=args.n_workers,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ffn_dim,
    )
