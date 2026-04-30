"""
Tune reconstruction hyperparameters on locally rotated validation patches.

The script first reconstructs the upright public patches to create a reference
map, then creates/uses a rotated validation dataset and scores several solver
settings by reconstructing that rotated set.
"""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from make_rotated_validation import create_rotated_validation
from reconstruction_solver import (
    SolverConfig,
    build_compatibility,
    estimate_overlap_width,
    grid_seam_energy,
    load_patches,
    reconstruct_grid,
    stitch_map,
)
from solution import choose_reconstruction_config


def image_array(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def mae_psnr(candidate_path: Path, reference_path: Path) -> tuple[float, float]:
    candidate = image_array(candidate_path)
    reference = image_array(reference_path)
    h = min(candidate.shape[0], reference.shape[0])
    w = min(candidate.shape[1], reference.shape[1])
    candidate = candidate[:h, :w]
    reference = reference[:h, :w]
    diff = candidate - reference
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return mae, psnr


def build_reference(base_dir: Path, output_dir: Path, force: bool) -> tuple[Path, int, int]:
    reference_path = output_dir / "reference_map.png"
    reference_cfg_path = output_dir / "reference_config.txt"
    patches = load_patches(base_dir / "patches")
    grid_size = int(round(len(patches) ** 0.5))
    if grid_size * grid_size != len(patches):
        raise ValueError(f"Patch count is not square: {len(patches)}")

    if reference_path.exists() and not force:
        overlap_width = estimate_overlap_width(patches, rotations=(0,))
        return reference_path, grid_size, overlap_width

    cfg = choose_reconstruction_config(
        patches=patches,
        grid_size=grid_size,
        selected_mode="none",
        patches_dir=base_dir / "patches",
    )
    grid = reconstruct_grid(patches, grid_size, cfg)
    stitch_map(patches, grid, reference_path, overlap_width=cfg.overlap_width)
    reference_cfg_path.write_text(
        f"overlap_width={cfg.overlap_width}\n"
        f"beam_width={cfg.beam_width}\n"
        f"beam_candidates={cfg.beam_candidates}\n",
        encoding="utf-8",
    )
    return reference_path, grid_size, int(cfg.overlap_width or 0)


def candidate_configs(base_cfg: SolverConfig, quick: bool, max_configs: int | None) -> list[SolverConfig]:
    beam_widths = (64, 96) if quick else (64, 96, 128, 192)
    beam_candidates = (12, 16) if quick else (12, 16, 24, 32)
    rotation_penalties = (0.0, 0.01, 0.03, 0.05)
    configs = []
    for beam_width in beam_widths:
        for candidate_count in beam_candidates:
            for penalty in rotation_penalties:
                configs.append(
                    replace(
                        base_cfg,
                        beam_width=beam_width,
                        beam_candidates=candidate_count,
                        refine_passes=2,
                        refine_candidates=12,
                        swap_refine_passes=5,
                        block_refine_passes=1,
                        rotations=(0, 1, 2, 3),
                        anchor_rotations=(0,),
                        rotation_penalty=penalty,
                    )
                )
                if max_configs is not None and len(configs) >= max_configs:
                    return configs
    return configs


def tune(
    base_dir: Path,
    validation_dir: Path,
    output_dir: Path,
    quick: bool,
    force: bool,
    max_configs: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path, grid_size, overlap_width = build_reference(base_dir, output_dir, force=force)

    if not (validation_dir / "patches").exists() or force:
        create_rotated_validation(
            patches_dir=base_dir / "patches",
            output_dir=validation_dir,
            seed=769,
            mode="random",
            include_rotation_tags=False,
        )

    patches = load_patches(validation_dir / "patches")
    base_cfg = SolverConfig(
        overlap_width=overlap_width,
        edge_width=1,
        gradient_weight=0.0,
        boundary_weight=0.8,
        rotations=(0, 1, 2, 3),
        anchor_rotations=(0,),
    )

    rows = []
    best = None
    for idx, cfg in enumerate(candidate_configs(base_cfg, quick=quick, max_configs=max_configs), start=1):
        print(
            f"[tune] {idx:02d} | beam={cfg.beam_width} | candidates={cfg.beam_candidates} | "
            f"rot_penalty={cfg.rotation_penalty}"
        )
        grid = reconstruct_grid(patches, grid_size, cfg)
        comp = build_compatibility(patches, cfg)
        seam = grid_seam_energy(grid, comp)
        candidate_path = output_dir / f"candidate_{idx:02d}.png"
        stitch_map(patches, grid, candidate_path, overlap_width=cfg.overlap_width)
        mae, psnr = mae_psnr(candidate_path, reference_path)
        row = {
            "idx": idx,
            "mae": mae,
            "psnr": psnr,
            "seam": seam,
            "overlap_width": cfg.overlap_width,
            "beam_width": cfg.beam_width,
            "beam_candidates": cfg.beam_candidates,
            "rotation_penalty": cfg.rotation_penalty,
            "refine_passes": cfg.refine_passes,
            "swap_refine_passes": cfg.swap_refine_passes,
            "block_refine_passes": cfg.block_refine_passes,
        }
        rows.append(row)
        if best is None or (mae, seam) < (best["mae"], best["seam"]):
            best = row
            shutil.copyfile(candidate_path, output_dir / "best_reconstructed_map.png")
        print(f"[score] mae={mae:.3f} | psnr={psnr:.2f} | seam={seam:.4f}")

    assert best is not None
    csv_path = output_dir / "tuning_results.csv"
    header = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(header) + "\n")
        for row in sorted(rows, key=lambda item: (item["mae"], item["seam"])):
            f.write(",".join(str(row[key]) for key in header) + "\n")

    print("[best]")
    for key, value in best.items():
        print(f"  {key}: {value}")
    print(f"[done] Results -> {csv_path}")
    print(f"[done] Best map -> {output_dir / 'best_reconstructed_map.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune reconstruction on rotated validation patches")
    parser.add_argument("base_dir", nargs="?", default=".", help="Project directory containing patches/")
    parser.add_argument("--validation-dir", default="rotated_validation", help="Rotated validation directory")
    parser.add_argument("--output-dir", default="tuning_outputs", help="Directory for tuning outputs")
    parser.add_argument("--full", action="store_true", help="Try a larger hyperparameter grid")
    parser.add_argument("--max-configs", type=int, default=4, help="Maximum configs to try; use 0 for no cap")
    parser.add_argument("--force", action="store_true", help="Regenerate validation/reference outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    tune(
        base_dir=base_dir,
        validation_dir=base_dir / args.validation_dir,
        output_dir=base_dir / args.output_dir,
        quick=not args.full,
        force=args.force,
        max_configs=None if args.max_configs == 0 else args.max_configs,
    )


if __name__ == "__main__":
    main()
