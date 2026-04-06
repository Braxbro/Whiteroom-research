"""
Fix Stage 9 checkpoints to include model_type field in config.

This script loads existing Stage 9 checkpoints, adds "model_type": "3stage" to the config,
and saves them back.
"""

import torch
from pathlib import Path

seeds = [1, 2, 3, 4, 5]
stage9_dir = Path("_agent/cache/runs/stage9")

for seed in seeds:
    ckpt_path = stage9_dir / f"stage9-seed{seed}" / "checkpoint_final.pt"

    if not ckpt_path.exists():
        print(f"Seed {seed}: checkpoint not found at {ckpt_path}")
        continue

    print(f"Seed {seed}: loading {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Add model_type if missing
    if "model_type" not in ckpt["config"]:
        ckpt["config"]["model_type"] = "3stage"
        torch.save(ckpt, ckpt_path)
        print(f"  ✓ Added model_type='3stage' and saved")
    else:
        print(f"  ✓ Already has model_type={ckpt['config']['model_type']}")

print("\nAll checkpoints fixed!")
