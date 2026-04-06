"""
Span freeze predictor — the "sibling model".

Two input formats:

  Full format (default):
    [old_tokens | SEP | new_tokens]   (~26 tokens for flag-append task)
    The extra flag token is the last token of new_tokens.

  Compact format (--compact):
    [old_tokens | SEP | pos_tok | new_tok]   (~15 tokens for flag-append task)
    Describes the edit as (position, new_value) using the existing PORT_IDX_0..9
    tokens for position. Generalizes to any single-token edit anywhere in the
    sequence — not just appends. FLOPs ~0.77x encoder recompute (vs 1.33x full).

    For flag-append: pos_tok = PORT_IDX_{b_end}, new_tok = extra_flag_tok.
    For arbitrary edits: pos_tok encodes the edit position, new_tok the new value.

Output:
    3 logits → sigmoid → (p_freeze_A, p_freeze_BIND, p_freeze_B)

At inference, round to binary (threshold 0.5).
"""

from __future__ import annotations
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import Token, VOCAB_SIZE_BASE, edit_pos_id, sibling_vocab_size
from .generator import VOCAB_SIZE
from .span_oracle import OracleSample, generate_oracle_dataset

# Compact sibling vocab size: primary VOCAB_SIZE + 32 edit position tokens
COMPACT_VOCAB_SIZE = sibling_vocab_size(VOCAB_SIZE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SpanFreezePredictor(nn.Module):
    """
    Small transformer encoder → 3-way span freeze predictor.

    d_model=64, 2 encoder layers, 4 heads is sufficient for sequences of
    length ~15 over a vocab of ~70 tokens.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        max_len: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=int(Token.PAD))
        self.pos_enc = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 3)  # 3 spans

    def forward(
        self,
        tokens: torch.Tensor,           # (batch, seq_len)
        key_padding_mask: Optional[torch.Tensor] = None,  # (batch, seq_len) bool, True=ignore
    ) -> torch.Tensor:
        """Returns logits of shape (batch, 3)."""
        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)
        x = self.embed(tokens) + self.pos_enc(positions)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        # Mean pool over non-padding positions
        if key_padding_mask is not None:
            mask = (~key_padding_mask).float().unsqueeze(-1)  # (B, L, 1)
            x = (x * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            x = x.mean(1)
        return self.head(x)  # (batch, 3)

    def predict(self, tokens: torch.Tensor) -> Tuple[int, int, int]:
        """Predict freeze combo for a single example (no batch dim)."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(tokens.unsqueeze(0))
            probs = torch.sigmoid(logits[0])
            fa, fb_ind, fb = (probs > 0.5).int().tolist()
        return fa, fb_ind, fb


# ---------------------------------------------------------------------------
# Dataset collation
# ---------------------------------------------------------------------------

def collate_samples(
    samples: List[OracleSample],
    device: torch.device,
    compact: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        tokens     (B, max_len) — padded input sequences
        pad_mask   (B, max_len) — True at padding positions
        labels     (B, 3)       — float targets for BCE loss

    compact=False: [old | SEP | new]          (~26 tokens)
    compact=True:  [old | SEP | extra_flag]   (~14 tokens)
    """
    sep = int(Token.SEP)
    pad = int(Token.PAD)
    seqs = []
    for s in samples:
        if compact:
            # Edit described as (position, new_value):
            # pos_tok = edit_pos_id(len(old_tokens)) — appended after primary vocab
            # new_tok = extra_flag_tok
            pos_tok = edit_pos_id(len(s.old_tokens), VOCAB_SIZE)
            seqs.append(s.old_tokens + [sep] + [pos_tok, s.extra_flag_tok])
        else:
            seqs.append(s.old_tokens + [sep] + s.new_tokens)
    max_len = max(len(s) for s in seqs)
    tokens_t = torch.full((len(seqs), max_len), pad, dtype=torch.long, device=device)
    pad_mask  = torch.ones((len(seqs), max_len), dtype=torch.bool, device=device)
    for i, seq in enumerate(seqs):
        tokens_t[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        pad_mask[i, :len(seq)] = False

    labels = torch.tensor(
        [list(s.span_combo) for s in samples],
        dtype=torch.float, device=device,
    )  # (B, 3)
    return tokens_t, pad_mask, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    checkpoint_path: str,
    output_path: str,
    n_samples: int = 2000,
    val_frac: float = 0.15,
    steps: int = 2000,
    batch_size: int = 64,
    lr: float = 3e-4,
    seed: int = 42,
    log_every: int = 100,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    ffn_dim: int = 256,
    compact: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Generate oracle dataset ---
    print(f"Generating oracle dataset ({n_samples} samples)...")
    samples = generate_oracle_dataset(
        checkpoint_path=checkpoint_path,
        n_samples=n_samples,
        seed=seed,
    )
    print(f"  Generated {len(samples)} samples")
    solvable = sum(s.has_accurate_mask for s in samples)
    print(f"  Solvable: {solvable}/{len(samples)} ({solvable/len(samples):.1%})")

    from collections import Counter
    combo_counts = Counter(s.span_combo for s in samples)
    print("  Combo distribution:")
    labels = {(1,1,1):"all",(0,0,0):"none",(1,0,0):"A",(0,1,0):"BIND",
              (0,0,1):"B",(1,1,0):"A+BIND",(1,0,1):"A+B",(0,1,1):"BIND+B"}
    for combo, count in combo_counts.most_common():
        print(f"    {labels.get(combo, str(combo))}: {count} ({count/len(samples):.1%})")

    # --- Split ---
    rng = random.Random(seed)
    rng.shuffle(samples)
    n_val = max(1, int(len(samples) * val_frac))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    # --- Model ---
    print(f"Input format: {'compact [old|SEP|pos|flag]' if compact else 'full [old|SEP|new]'}")
    model_vocab = COMPACT_VOCAB_SIZE if compact else VOCAB_SIZE
    model = SpanFreezePredictor(
        vocab_size=model_vocab,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    log_rows = []

    # --- Training loop ---
    model.train()
    for step in range(1, steps + 1):
        batch = rng.choices(train_samples, k=batch_size)
        tokens, pad_mask, labels_t = collate_samples(batch, device, compact=compact)
        logits = model(tokens, pad_mask)
        loss = F.binary_cross_entropy_with_logits(logits, labels_t)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if step % log_every == 0:
            # Validation
            model.eval()
            with torch.no_grad():
                vtok, vmask, vlabels = collate_samples(val_samples, device, compact=compact)
                vlogits = model(vtok, vmask)
                vloss = F.binary_cross_entropy_with_logits(vlogits, vlabels).item()
                vpreds = (torch.sigmoid(vlogits) > 0.5).int()
                vlabels_int = vlabels.int()
                # Exact combo match
                combo_acc = (vpreds == vlabels_int).all(dim=1).float().mean().item()
                # Per-span accuracy
                span_acc = (vpreds == vlabels_int).float().mean(dim=0).tolist()
            model.train()

            row = {
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(vloss, 4),
                "val_combo_acc": round(combo_acc, 4),
                "val_span_acc_A": round(span_acc[0], 4),
                "val_span_acc_BIND": round(span_acc[1], 4),
                "val_span_acc_B": round(span_acc[2], 4),
            }
            log_rows.append(row)
            print(
                f"step {step:4d} | loss {loss.item():.4f} | val_loss {vloss:.4f} | "
                f"combo_acc {combo_acc:.3f} | "
                f"span_acc A={span_acc[0]:.3f} BIND={span_acc[1]:.3f} B={span_acc[2]:.3f}"
            )

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "vocab_size": COMPACT_VOCAB_SIZE if compact else VOCAB_SIZE,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "ffn_dim": ffn_dim,
        },
        "compact": compact,
        "val_combo_acc": log_rows[-1]["val_combo_acc"] if log_rows else None,
    }, output_path)
    print(f"\nSaved to {output_path}")

    log_path = output_path.replace(".pt", "_log.jsonl")
    with open(log_path, "w") as f:
        for row in log_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Log: {log_path}")
    return log_rows


# ---------------------------------------------------------------------------
# Evaluation: downstream compound accuracy with predicted masks
# ---------------------------------------------------------------------------

def evaluate_downstream(
    sibling_checkpoint: str,
    primary_checkpoint: str,
    n_pairs: int = 300,
    seed: int = 99,
) -> dict:
    """
    Evaluate downstream flag accuracy when using the sibling's predicted mask
    vs oracle mask vs freeze-all vs full-fresh.
    """
    from .model import WhiteroomTransformer
    from .span_oracle import run_span_oracle, build_hybrid
    from .freeze_probe import make_example_for_ab, _greedy_from_memory, FLAG_TOKENS, _flag_tok
    from .generator import sample_primitive
    from .composition import find_valid_bindings
    from .vocab import TRAINING_FLAGS, flag_token, Flag

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load primary model
    ckpt_p = torch.load(primary_checkpoint, map_location=device)
    primary = WhiteroomTransformer(**ckpt_p["config"]).to(device)
    primary.load_state_dict(ckpt_p["model_state"])
    primary.eval()

    # Load sibling model
    ckpt_s = torch.load(sibling_checkpoint, map_location=device, weights_only=False)
    sibling = SpanFreezePredictor(**ckpt_s["config"]).to(device)
    sibling.load_state_dict(ckpt_s["model_state"])
    sibling.eval()
    sibling_compact = ckpt_s.get("compact", False)

    rng = random.Random(seed)

    metrics = {k: 0 for k in ("freeze_all", "full_fresh", "oracle", "sibling")}
    n_total = 0

    sep = int(Token.SEP)

    for _ in range(n_pairs):
        for _ in range(50):
            a = sample_primitive(rng)
            b = sample_primitive(rng)
            bindings = find_valid_bindings(a, b)
            if bindings:
                break
        else:
            continue

        port_a_idx, port_b_idx = rng.choice(bindings)
        combined = a.flags | b.flags
        available = [f for f in TRAINING_FLAGS if f not in combined]
        if not available:
            continue
        extra_flag = rng.choice(available)

        oracle_result = run_span_oracle(primary, a, b, port_a_idx, port_b_idx, extra_flag, device)

        ex = make_example_for_ab(a, b, port_a_idx, port_b_idx)
        extra_tok = flag_token(extra_flag)
        base_tokens = ex.input_tokens
        ext_tokens = base_tokens + [extra_tok]
        L = len(base_tokens)
        a_s, a_e = ex.a_token_span
        b_s, b_e = ex.b_token_span

        target_flags = frozenset(_flag_tok(f) for f in (a.flags | b.flags | {extra_flag}))

        src_old = torch.tensor(base_tokens, dtype=torch.long, device=device).unsqueeze(0)
        src_new = torch.tensor(ext_tokens,  dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            mem_old = primary.encode(src_old)
            mem_new = primary.encode(src_new)

        def decode_with_mask(mask):
            hybrid = build_hybrid(mem_old, mem_new, mask)
            with torch.no_grad():
                pred = _greedy_from_memory(primary, hybrid, device, 32)
            pred_flags = frozenset(t for t in pred if t in FLAG_TOKENS)
            return pred_flags == target_flags

        # freeze_all baseline
        metrics["freeze_all"] += decode_with_mask([1] * L)

        # full_fresh baseline
        metrics["full_fresh"] += decode_with_mask([0] * L)

        # oracle mask
        if oracle_result.optimal_combo is not None:
            fa, fb_ind, fb = oracle_result.optimal_combo
        else:
            fa, fb_ind, fb = 1, 1, 1
        oracle_mask = [0] * L
        if fa:
            for i in range(a_s, a_e): oracle_mask[i] = 1
        if fb_ind:
            for i in range(a_e, b_s): oracle_mask[i] = 1
        if fb:
            for i in range(b_s, b_e): oracle_mask[i] = 1
        metrics["oracle"] += decode_with_mask(oracle_mask)

        # sibling prediction
        if sibling_compact:
            pos_tok = edit_pos_id(len(base_tokens), VOCAB_SIZE)
            sib_seq = base_tokens + [sep] + [pos_tok, extra_tok]
        else:
            sib_seq = base_tokens + [sep] + ext_tokens
        sibling_input = torch.tensor(sib_seq, dtype=torch.long, device=device)
        sfa, sfb_ind, sfb = sibling.predict(sibling_input)
        sib_mask = [0] * L
        if sfa:
            for i in range(a_s, a_e): sib_mask[i] = 1
        if sfb_ind:
            for i in range(a_e, b_s): sib_mask[i] = 1
        if sfb:
            for i in range(b_s, b_e): sib_mask[i] = 1
        metrics["sibling"] += decode_with_mask(sib_mask)

        n_total += 1

    return {k: v / n_total for k, v in metrics.items()} | {"n": n_total}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    t = sub.add_parser("train")
    t.add_argument("--primary",   required=True)
    t.add_argument("--output",    required=True)
    t.add_argument("--n-samples", type=int, default=2000)
    t.add_argument("--steps",     type=int, default=2000)
    t.add_argument("--batch-size",type=int, default=64)
    t.add_argument("--lr",        type=float, default=3e-4)
    t.add_argument("--seed",      type=int, default=42)
    t.add_argument("--compact",   action="store_true",
                   help="Use compact input [old|SEP|flag] instead of [old|SEP|new]")

    e = sub.add_parser("eval")
    e.add_argument("--sibling", required=True)
    e.add_argument("--primary", required=True)
    e.add_argument("--n-pairs", type=int, default=300)

    args = parser.parse_args()

    if args.cmd == "train":
        train(
            checkpoint_path=args.primary,
            output_path=args.output,
            n_samples=args.n_samples,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            compact=args.compact,
        )
    elif args.cmd == "eval":
        results = evaluate_downstream(args.sibling, args.primary, args.n_pairs)
        print("\nDownstream flag accuracy:")
        for k, v in results.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
