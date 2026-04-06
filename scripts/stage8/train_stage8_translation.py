"""
Stage 8: Translation Layer Experiment

Combines a frozen 7d encoder (block-diagonal isolation, perfect -0.0020 B_frozen_deg)
with a frozen Stage5 decoder (best composition recovery, 77%) via a learned projection
layer. Only the projection trains. Goal: deliberately recreate Stage 4b Seed 4's jackpot
performance through architectural design rather than RNG luck.

The projection bridges 7d's block-diagonal representation space to the space Stage5's
decoder was trained on.

Usage:
    python -m _agent.scripts.stage8.train_stage8_translation \\
        --encoder-ckpt _agent/cache/runs/stage7/7d-sawtooth/stage5-seed1/checkpoint_final.pt \\
        --decoder-ckpt _agent/cache/runs/stage5/stage5-seed1/checkpoint_final.pt \\
        --checkpoint-dir _agent/cache/runs/stage8/7d-seed1_stage5-seed1 \\
        --seed 1
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from whiteroom.generator import VOCAB_SIZE, balanced_archetype_weights
from whiteroom.model import WhiteroomTransformer
from whiteroom.vocab import Token
from whiteroom.finetune_curriculum import collate_curriculum, CurriculumPrefetcher
from whiteroom.train import collate, collate_attribution, compute_loss, DataPrefetcher


# =============================================================================
# Core Components
# =============================================================================

class TranslationProjection(nn.Module):
    """LayerNorm(input_dim) + Linear(input_dim→output_dim) with bias.

    Maps 7d encoder representation space to Stage5 decoder expectation space.
    Both use d_model=64 (363K architecture), so this is LayerNorm(64) + Linear(64→64).
    Only this layer trains; encoder and decoder freeze.
    """
    def __init__(self, d_in: int = 64, d_out: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.linear = nn.Linear(d_in, d_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) or (batch, d_model)"""
        return self.linear(self.norm(x))


class MLPProjection(nn.Module):
    """LayerNorm + 2-layer MLP (expand-then-contract).

    Maps 7d encoder representation space to Stage5 decoder expectation space
    via a wider intermediate layer. Structure:
      LayerNorm(d_in) → Linear(d_in→4*d_in) → ReLU → Linear(4*d_in→d_out)

    For d_in=d_out=64, this is ~4*64^2 ≈ 16K params vs 4K for linear.
    Adds capacity for learning nonlinear alignment between representation spaces.
    """
    def __init__(self, d_in: int = 64, d_out: int = 64, hidden_mult: int = 4):
        super().__init__()
        hidden_dim = d_in * hidden_mult
        self.norm = nn.LayerNorm(d_in)
        self.fc1 = nn.Linear(d_in, hidden_dim, bias=True)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, d_out, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model) or (batch, d_model)"""
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def load_checkpoint(path: str, device: torch.device) -> WhiteroomTransformer:
    """Load a WhiteroomTransformer checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = WhiteroomTransformer(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model


def forward_decoder(
    decoder: WhiteroomTransformer,
    projected_mem: torch.Tensor,
    tgt_in: torch.Tensor,
    src_pad_mask: Optional[torch.Tensor] = None,
    tgt_pad_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run frozen Stage5 decoder on projected memory.

    decoder stays in train() mode to enable dropout for numerical stability.
    Gradients flow backward through the decoder's forward pass to the projection,
    while requires_grad_(False) prevents decoder parameter updates.

    Returns: (seq_logits, valid_logits)
    """
    tgt_len = tgt_in.size(1)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt_in.device)

    dec_out = decoder.decode(
        tgt_in, projected_mem,
        tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_pad_mask,
        memory_key_padding_mask=src_pad_mask,
    )

    seq_logits = decoder.seq_head(dec_out)  # (batch, tgt_len, vocab_size)

    # valid_logits: mean-pool projected memory (NOT frozen positions)
    if src_pad_mask is not None:
        not_pad = (~src_pad_mask).unsqueeze(-1).float()
        pooled = (projected_mem * not_pad).sum(dim=1) / not_pad.sum(dim=1).clamp(min=1)
    else:
        pooled = projected_mem.mean(dim=1)
    valid_logits = decoder.valid_head(pooled)  # (batch, 1)

    return seq_logits, valid_logits


def save_checkpoint(
    path: str,
    step: int,
    phase: int,
    phase_step: int,
    projection: TranslationProjection,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    encoder_ckpt: str,
    decoder_ckpt: str,
    encoder_ckpt_data: dict,
    decoder_ckpt_data: dict,
    train_config: dict,
    note: str = "",
):
    """Save checkpoint_translation.pt with projection state and metadata."""
    torch.save({
        "step": step,
        "phase": phase,
        "phase_step": phase_step,
        "projection_state": projection.state_dict(),
        "encoder_ckpt": encoder_ckpt,
        "decoder_ckpt": decoder_ckpt,
        "encoder_config": encoder_ckpt_data.get("config", {}),
        "decoder_config": decoder_ckpt_data.get("config", {}),
        "projection_config": {
            "d_in": projection.norm.weight.shape[0],
            "d_out": projection.fc2.weight.shape[0] if hasattr(projection, 'fc2') else projection.linear.weight.shape[0],
            "type": "mlp" if hasattr(projection, 'fc2') else "linear",
        },
        "train_config": train_config,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "note": note,
    }, path)


# =============================================================================
# Training Loop
# =============================================================================

def train_stage8(
    encoder_ckpt: str,
    decoder_ckpt: str,
    checkpoint_dir: str,
    seed: int = 1,
    # Training params (same as Stage5)
    steps: int = 50_000,
    batch_size: int = 64,
    lr: float = 3e-4,
    curriculum_prob: float = 0.4,
    n_workers: int = 3,
    balance_archetypes: bool = True,
    cooccurrence_damp: float = 0.7,
    log_every: int = 100,
    checkpoint_every: int = 2_000,
    # Plateau detection (same as Stage5)
    plateau_window: int = 10,
    plateau_threshold: float = 2e-4,
    min_phase_steps: int = 2_000,  # Projection converges faster than full models
    force_phase: Optional[int] = None,  # Override initial phase (1 or 2)
    resume_ckpt: Optional[str] = None,  # Resume from checkpoint
    projection_type: str = "linear",  # "linear" or "mlp"
    unfreeze_decoder: bool = False,  # Fine-tuning: unfreeze decoder for training
):
    import random
    rng = random.Random(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Stage 8 — Translation Layer")
    print(f"Seed: {seed}")
    print(f"Encoder: {encoder_ckpt}")
    print(f"Decoder: {decoder_ckpt}")
    print(f"Curriculum prob: {curriculum_prob}, batch_size: {batch_size}, lr: {lr}")

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "train_log.jsonl"
    run_log_path = ckpt_dir / "run_log.txt"

    # --- Load frozen models ---
    print("\nLoading encoder (7d)...")
    encoder_ckpt_data = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)
    encoder_config = encoder_ckpt_data["config"].copy()
    # Handle rename: sawtooth_encoder → block_diag_encoder_mask
    if "sawtooth_encoder" in encoder_config:
        encoder_config["block_diag_encoder_mask"] = encoder_config.pop("sawtooth_encoder")
    encoder = WhiteroomTransformer(**encoder_config).to(device)
    encoder.load_state_dict(encoder_ckpt_data["model_state"])
    encoder.requires_grad_(False)
    # IMPORTANT: Keep encoder in train() mode (not eval) to enable dropout
    # This prevents NaN overflow in attention on long sequences (e.g., attribution examples)
    # Dropout acts as regularization and stabilizes attention logits during inference
    # requires_grad_(False) prevents parameter updates; train() only controls dropout/batchnorm
    encoder.train()

    print("Loading decoder (Stage5)...")
    decoder_ckpt_data = torch.load(decoder_ckpt, map_location="cpu", weights_only=False)
    decoder_config = decoder_ckpt_data["config"].copy()
    # Handle rename: sawtooth_encoder → block_diag_encoder_mask
    if "sawtooth_encoder" in decoder_config:
        decoder_config["block_diag_encoder_mask"] = decoder_config.pop("sawtooth_encoder")
    decoder = WhiteroomTransformer(**decoder_config).to(device)
    decoder.load_state_dict(decoder_ckpt_data["model_state"])

    if unfreeze_decoder:
        # Fine-tuning mode: decoder is trainable
        decoder.requires_grad_(True)
        decoder.train()
        print(f"Decoder unfrozen for fine-tuning")
    else:
        # Standard Stage 8: decoder frozen
        decoder.requires_grad_(False)
        # IMPORTANT: Keep decoder in train() mode (not eval) to enable dropout
        # Prevents NaN overflow in attention during decoding
        # requires_grad_(False) prevents parameter updates; train() only controls dropout/batchnorm
        decoder.train()

    print("Creating projection...")
    # Projection maps encoder output (d_in) to decoder input (d_out)
    # Both are d_model=64 for the 363K architecture
    encoder_d_model = encoder.d_model  # 64 for 363K 7d
    decoder_d_model = decoder.d_model  # 64 for 363K Stage5

    if projection_type == "linear":
        projection = TranslationProjection(d_in=encoder_d_model, d_out=decoder_d_model).to(device)
    elif projection_type == "mlp":
        projection = MLPProjection(d_in=encoder_d_model, d_out=decoder_d_model).to(device)
    else:
        raise ValueError(f"Unknown projection_type: {projection_type}")

    projection.train()

    n_params_proj = sum(p.numel() for p in projection.parameters())
    print(f"Projection parameters: {n_params_proj:,}")

    # Verify freezing
    assert all(not p.requires_grad for p in encoder.parameters()), "Encoder must be frozen"
    if not unfreeze_decoder:
        assert all(not p.requires_grad for p in decoder.parameters()), "Decoder must be frozen"
    assert all(p.requires_grad for p in projection.parameters()), "Projection must be trainable"

    # --- Optimizer & scheduler ---
    # Collect trainable parameters
    trainable_params = list(projection.parameters())
    if unfreeze_decoder:
        trainable_params.extend(decoder.parameters())

    # Use specified lr for all trainable params during fine-tuning, or proj_lr for projection-only
    if unfreeze_decoder:
        optimizer_lr = lr  # Use standard lr for fine-tuning
        print(f"Fine-tuning mode: projection + decoder LR: {optimizer_lr}")
    else:
        optimizer_lr = 1e-4  # Projection-only: lower LR for bridging
        print(f"Projection-only mode: projection LR: {optimizer_lr} (overridden from {lr})")

    optimizer = torch.optim.Adam(trainable_params, lr=optimizer_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)

    # --- Data ---
    arch_weights = balanced_archetype_weights() if balance_archetypes else None
    prefetcher = DataPrefetcher(
        base_seed=seed,
        n_workers=n_workers,
        balance_archetypes=balance_archetypes,
        cooccurrence_damp=cooccurrence_damp,
    )
    curr_prefetcher = CurriculumPrefetcher(
        base_seed=seed,
        n_workers=n_workers,
        weights=arch_weights,
        cooccurrence_damp=cooccurrence_damp,
    )

    def _flip_comp_example(ex):
        """Flip A↔B in composition example for bidirectional BIND."""
        # Not needed for stage8 (unidirectional BIND)
        return ex

    def _get_comp(n):
        items = prefetcher.get_comp(n)
        return items

    def _get_attr(n):
        items = prefetcher.get_attr(n)
        return items

    def _get_curr(n):
        items = curr_prefetcher.get(n)
        return items

    # --- Loss functions ---
    seq_loss_fn = nn.CrossEntropyLoss(ignore_index=Token.PAD, reduction="mean")
    valid_loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    # --- Phase tracking ---
    phase = force_phase if force_phase is not None else 1
    phase_step = 0
    curr_loss_history = []
    phase1_transition_step = None

    if force_phase is not None:
        print(f"Force starting in phase {force_phase}")
        if force_phase == 2:
            phase1_transition_step = 0  # Mark transition as occurred

    running = {"seq": 0.0, "valid": 0.0, "attr": 0.0, "curr": 0.0, "total": 0.0}

    # --- Resume from checkpoint if provided ---
    start_step = 1
    if resume_ckpt is not None:
        _print = print
        _print(f"Resuming from checkpoint: {resume_ckpt}")
        ckpt_data = torch.load(resume_ckpt, map_location=device, weights_only=False)

        start_step = ckpt_data["step"] + 1
        phase = force_phase if force_phase is not None else ckpt_data["phase"]
        phase_step = ckpt_data.get("phase_step", 0)

        # Load projection state
        projection.load_state_dict(ckpt_data["projection_state"])

        # Load optimizer and scheduler state (skip if unfreezing decoder — parameter groups changed)
        if not unfreeze_decoder:
            optimizer.load_state_dict(ckpt_data["optimizer_state"])
            scheduler.load_state_dict(ckpt_data["scheduler_state"])
        else:
            _print("Skipping optimizer/scheduler state load (decoder unfrozen — parameter groups changed)")

        if force_phase is not None:
            _print(f"Overriding phase to {force_phase}")
            phase = force_phase
            if force_phase == 2:
                phase1_transition_step = 0

        _print(f"Resuming at step {start_step}, phase {phase}")

    t0 = time.time()

    # --- Training loop ---
    with open(run_log_path, "w") as run_log:
        def _log(msg):
            print(msg)
            run_log.write(msg + "\n")
            run_log.flush()

        for step in range(start_step, steps + 1):
            projection.train()
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            components = {k: 0.0 for k in running}

            if rng.random() < curriculum_prob:
                # --- Curriculum batch ---
                samples = _get_curr(batch_size)
                partial = (phase == 1)
                # Pass encoder directly to collate_curriculum
                hybrid_mem, tgt_in, tgt_out = collate_curriculum(
                    samples, device, encoder, rng=rng, partial_freeze=partial)

                # Apply projection AFTER collate_curriculum (so gradients flow)
                projected_mem = projection(hybrid_mem)

                # Decode (frozen, in eval mode, not wrapped in no_grad)
                seq_logits, valid_logits = forward_decoder(
                    decoder, projected_mem, tgt_in,
                    tgt_pad_mask=(tgt_in == Token.PAD),
                )

                b, t, v = seq_logits.shape
                curr_loss = seq_loss_fn(seq_logits.reshape(b * t, v), tgt_out.reshape(b * t))
                total_loss = total_loss + curr_loss
                components["curr"] = curr_loss.item()

            else:
                # --- Normal composition + attribution batch ---
                comp_size = batch_size // 2
                comp_examples = _get_comp(comp_size)
                comp_batch = collate(comp_examples, device)

                # Encode (frozen, in no_grad)
                with torch.no_grad():
                    mem = encoder.encode(comp_batch["src"], src_key_padding_mask=comp_batch["src_pad_mask"])

                # Project (gradients flow)
                projected_mem = projection(mem)

                # Decode (frozen, in train mode for dropout stability)
                seq_logits, valid_logits = forward_decoder(
                    decoder, projected_mem,
                    comp_batch["tgt_in"],
                    src_pad_mask=comp_batch["src_pad_mask"],
                    tgt_pad_mask=comp_batch["tgt_pad_mask"],
                )

                comp_loss, comp_components = compute_loss(
                    seq_logits, valid_logits,
                    comp_batch["tgt_out"], comp_batch["is_valid"],
                    seq_loss_fn, valid_loss_fn,
                )
                total_loss = total_loss + comp_loss
                components["seq"] = comp_components["seq"]
                components["valid"] = comp_components["valid"]

                # Attribution batch (not curriculum, so frozen positions not applicable)
                attr_examples = _get_attr(batch_size // 2)
                attr_batch = collate_attribution(attr_examples, device)

                with torch.no_grad():
                    mem = encoder.encode(attr_batch["src"], src_key_padding_mask=attr_batch["src_pad_mask"])

                projected_mem = projection(mem)

                seq_logits_attr, _ = forward_decoder(
                    decoder, projected_mem,
                    attr_batch["tgt_in"],
                    src_pad_mask=attr_batch["src_pad_mask"],
                    tgt_pad_mask=attr_batch["tgt_pad_mask"],
                )

                b, t, v = seq_logits_attr.shape
                tgt_attr_flat = attr_batch["tgt_out"].reshape(b * t)

                # Check for NaN/Inf (should be rare/nonexistent with dropout enabled)
                has_nan = torch.isnan(seq_logits_attr).any()
                has_inf = torch.isinf(seq_logits_attr).any()

                if has_nan or has_inf:
                    # Skip batches with NaN (graceful degradation)
                    components["attr"] = 0.0
                elif tgt_attr_flat.min() < 0 or tgt_attr_flat.max() >= v:
                    print(f"\nDEBUG at step {step}: Invalid attribution targets!")
                    print(f"  Target range: [{tgt_attr_flat.min()}, {tgt_attr_flat.max()}]")
                    print(f"  Valid range: [0, {v - 1}]")
                    components["attr"] = 0.0
                else:
                    attr_loss = seq_loss_fn(seq_logits_attr.reshape(b * t, v), tgt_attr_flat)
                    if torch.isnan(attr_loss):
                        print(f"\nDEBUG at step {step}: Loss computation produced NaN!")
                        print(f"  seq_logits_attr: min={seq_logits_attr.min():.6f}, max={seq_logits_attr.max():.6f}")
                        print(f"  targets: min={tgt_attr_flat.min()}, max={tgt_attr_flat.max()}")
                        components["attr"] = 0.0
                    else:
                        total_loss = total_loss + attr_loss
                        components["attr"] = attr_loss.item()

            # Backward + step
            total_loss.backward()
            nn.utils.clip_grad_norm_(projection.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Running average
            for k in running:
                running[k] = (running[k] * 0.95) + (components[k] * 0.05)
            components["total"] = total_loss.item()

            # NaN detection
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"\nERROR at step {step}: Loss is NaN/Inf!")
                print(f"Components: {components}")
                print(f"Stopping training.")
                break

            running["total"] = (running["total"] * 0.95) + (components["total"] * 0.05)

            # Logging
            if step % log_every == 0:
                elapsed = time.time() - t0
                rate = step / elapsed
                eta = (steps - step) / rate if rate > 0 else 0
                log_line = (
                    f"step {step:6d} [ph{phase}] | "
                    f"loss {running['total']:.4f} | "
                    f"seq {running['seq']:.4f} | "
                    f"valid {running['valid']:.4f} | "
                    f"attr {running['attr']:.4f} | "
                    f"curr {running['curr']:.4f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                    f"{elapsed:.0f}s"
                )
                _log(log_line)

                # Write to jsonl
                with open(log_path, "a") as f:
                    json.dump({
                        "step": step,
                        "phase": phase,
                        "seq": running["seq"],
                        "valid": running["valid"],
                        "attr": running["attr"],
                        "curr": running["curr"],
                        "total": running["total"],
                        "lr": optimizer.param_groups[0]["lr"],
                        "elapsed": elapsed,
                    }, f)
                    f.write("\n")

            # Plateau detection for curriculum loss
            if "curr" in components and components["curr"] > 0:
                curr_loss_history.append(components["curr"])

            phase_step += 1

            if len(curr_loss_history) > plateau_window:
                # Check if plateau
                window = curr_loss_history[-plateau_window:]
                x = list(range(len(window)))
                y = window
                # Linear regression: slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
                n = len(window)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(xi * yi for xi, yi in zip(x, y))
                sum_x2 = sum(xi * xi for xi in x)
                denom = n * sum_x2 - sum_x * sum_x
                slope = (n * sum_xy - sum_x * sum_y) / denom if denom != 0 else 0

                if phase_step >= min_phase_steps and abs(slope) < plateau_threshold:
                    if phase == 1:
                        _log(f"Phase 1 plateau detected at step {step}. Transitioning to Phase 2...")
                        phase1_transition_step = step
                        save_checkpoint(
                            ckpt_dir / "checkpoint_phase1_transition.pt",
                            step, phase, phase_step, projection, optimizer, scheduler,
                            encoder_ckpt, decoder_ckpt,
                            encoder_ckpt_data, decoder_ckpt_data,
                            dict(curriculum_prob=curriculum_prob, batch_size=batch_size, lr=lr),
                            note="phase1_plateau",
                        )
                        phase = 2
                        phase_step = 0
                        curr_loss_history = []
                    elif phase == 2:
                        _log(f"Phase 2 plateau detected at step {step}. Training complete.")
                        break

            # Checkpointing
            if step % checkpoint_every == 0:
                save_checkpoint(
                    ckpt_dir / f"checkpoint_{step:06d}.pt",
                    step, phase, phase_step, projection, optimizer, scheduler,
                    encoder_ckpt, decoder_ckpt,
                    encoder_ckpt_data, decoder_ckpt_data,
                    dict(curriculum_prob=curriculum_prob, batch_size=batch_size, lr=lr),
                    note="periodic",
                )

        # Final checkpoint (named checkpoint_translation.pt for eval compatibility)
        save_checkpoint(
            ckpt_dir / "checkpoint_translation.pt",
            step, phase, phase_step, projection, optimizer, scheduler,
            encoder_ckpt, decoder_ckpt,
            encoder_ckpt_data, decoder_ckpt_data,
            dict(curriculum_prob=curriculum_prob, batch_size=batch_size, lr=lr),
            note="final",
        )
        _log(f"Training complete. Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 8: Translation Layer Training")
    parser.add_argument("--encoder-ckpt", type=str, required=True, help="Path to 7d encoder checkpoint")
    parser.add_argument("--decoder-ckpt", type=str, required=True, help="Path to Stage5 decoder checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--steps", type=int, default=50_000, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--curriculum-prob", type=float, default=0.4, help="Curriculum batch probability")
    parser.add_argument("--n-workers", type=int, default=3, help="Data worker threads")
    parser.add_argument("--balance-archetypes", action="store_true", help="Balance archetype weights")
    parser.add_argument("--cooccurrence-damp", type=float, default=0.7, help="Cooccurrence damping")
    parser.add_argument("--log-every", type=int, default=100, help="Logging interval")
    parser.add_argument("--checkpoint-every", type=int, default=2_000, help="Checkpoint interval")
    parser.add_argument("--plateau-window", type=int, default=10, help="Plateau detection window")
    parser.add_argument("--plateau-threshold", type=float, default=2e-4, help="Plateau threshold")
    parser.add_argument("--min-phase-steps", type=int, default=2_000, help="Min steps before plateau can trigger")
    parser.add_argument("--force-phase", type=int, default=None, help="Force start in this phase (1 or 2) instead of auto-detecting")
    parser.add_argument("--resume-ckpt", type=str, default=None, help="Resume from a checkpoint (loads step/phase/optimizer state)")
    parser.add_argument("--projection-type", type=str, default="linear", choices=["linear", "mlp"], help="Projection type: 'linear' (LayerNorm+Linear) or 'mlp' (LayerNorm+MLP)")
    parser.add_argument("--unfreeze-decoder", action="store_true", help="Unfreeze decoder for fine-tuning (train projection + decoder)")

    args = parser.parse_args()
    train_stage8(**vars(args))
