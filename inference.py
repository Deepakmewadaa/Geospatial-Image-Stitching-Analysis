"""
Automated grading entrypoint.

Required by the evaluator:
    python inference.py --test_dir <absolute_path_to_test_dir>

Reads patches/ and test.csv from --test_dir, writes submission.csv in the
current working directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from reconstruction_solver import estimate_overlap_width, load_patches, reconstruct_grid, stitch_map
from solution import SolverConfig, answer_questions, load_model


os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project inference script")
    parser.add_argument("--test_dir", required=True, help="Absolute path to hidden test directory")
    parser.add_argument("--model-size", choices=("2b", "7b"), default="7b")
    parser.add_argument("--vqa-image-size", type=int, default=1024)
    parser.add_argument("--vqa-view-mode", choices=("full", "full-crops"), default="full-crops")
    parser.add_argument("--vqa-max-new-tokens", type=int, default=32)
    parser.add_argument("--vqa-prompt-style", choices=("strict", "reason-then-answer"), default="reason-then-answer")
    parser.add_argument("--beam-width", type=int, default=96)
    parser.add_argument("--beam-candidates", type=int, default=16)
    return parser.parse_args()


def reconstruct_from_test_dir(test_dir: Path, output_map: Path, args: argparse.Namespace):
    patches = load_patches(test_dir / "patches")
    n = len(patches)
    grid_size = int(round(n**0.5))
    if grid_size * grid_size != n:
        raise ValueError(f"Patch count is not square: {n}")

    overlap_width = estimate_overlap_width(patches, rotations=(0, 1, 2, 3))
    cfg = SolverConfig(
        overlap_width=overlap_width,
        edge_width=1,
        gradient_weight=0.0,
        boundary_weight=0.8,
        beam_width=args.beam_width,
        beam_candidates=args.beam_candidates,
        refine_passes=2,
        refine_candidates=12,
        swap_refine_passes=4,
        block_refine_passes=1,
        rotations=(0, 1, 2, 3),
        anchor_rotations=(0,),
        rotation_penalty=0.01,
    )
    print(
        f"[infer] Reconstruction config | overlap={overlap_width} | "
        f"beam={cfg.beam_width} | candidates={cfg.beam_candidates}"
    )
    grid = reconstruct_grid(patches, grid_size, cfg)
    return stitch_map(patches, grid, output_map, overlap_width=overlap_width)


def main() -> None:
    args = parse_args()
    test_dir = Path(args.test_dir).resolve()
    cwd = Path.cwd()
    output_csv = cwd / "submission.csv"
    output_map = cwd / "reconstructed_map.png"

    if not test_dir.exists():
        raise FileNotFoundError(f"test_dir not found: {test_dir}")
    if not (test_dir / "patches").exists():
        raise FileNotFoundError(f"patches folder not found in {test_dir}")
    if not (test_dir / "test.csv").exists():
        raise FileNotFoundError(f"test.csv not found in {test_dir}")

    print(f"[infer] test_dir={test_dir}")
    map_img = reconstruct_from_test_dir(test_dir, output_map, args)

    model_name = "Qwen2-VL-2B-Instruct" if args.model_size == "2b" else "Qwen2-VL-7B-Instruct"
    model_dir = cwd / "models" / model_name
    model, processor, process_vision_info, torch = load_model(model_dir)

    test_df = pd.read_csv(test_dir / "test.csv")
    print(f"[infer] Loaded {len(test_df)} questions from {test_dir / 'test.csv'}")
    submission_df = answer_questions(model, processor, process_vision_info, torch, map_img, test_df, args)
    submission_df.to_csv(output_csv, index=False)
    print(f"[done] Submission -> {output_csv}")


if __name__ == "__main__":
    main()
