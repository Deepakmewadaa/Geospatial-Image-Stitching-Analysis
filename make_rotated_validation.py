"""
Create a hidden-test-style rotated patch validation set.

The public patches in this project are upright. The hidden test can rotate
patches by multiples of 90 degrees while keeping patch_0 fixed as the top-left
anchor. This script creates that same condition locally so reconstruction can be
validated before submission.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from PIL import Image


ROTATION_DEGREES = (0, 90, 180, 270)


def patch_id(path: Path) -> int:
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Expected patch_N.png name, got {path.name}") from exc


def iter_patch_paths(patches_dir: Path) -> list[Path]:
    paths = sorted(patches_dir.glob("patch_*.png"), key=patch_id)
    if not paths:
        raise FileNotFoundError(f"No patch_*.png files found in {patches_dir}")
    return paths


def choose_rotation(pid: int, mode: str, rng: random.Random) -> int:
    if pid == 0:
        return 0
    if mode == "random":
        return rng.choice(ROTATION_DEGREES)
    if mode == "random-nonzero":
        return rng.choice(ROTATION_DEGREES[1:])
    if mode == "cycle":
        return ROTATION_DEGREES[pid % len(ROTATION_DEGREES)]
    raise ValueError(f"Unknown rotation mode: {mode}")


def rotate_image(img: Image.Image, degrees_clockwise: int) -> Image.Image:
    # PIL rotates counter-clockwise, so negate to create clockwise test rotations.
    return img.rotate(-degrees_clockwise, expand=True)


def create_rotated_validation(
    patches_dir: Path,
    output_dir: Path,
    seed: int,
    mode: str,
    include_rotation_tags: bool,
) -> None:
    output_patches = output_dir / "patches"
    output_patches.mkdir(parents=True, exist_ok=True)
    for old_patch in output_patches.glob("patch_*.png"):
        old_patch.unlink()

    rows = []
    rng = random.Random(seed)
    for src in iter_patch_paths(patches_dir):
        pid = patch_id(src)
        degrees = choose_rotation(pid, mode, rng)
        with Image.open(src).convert("RGB") as img:
            rotated = rotate_image(img, degrees)
            if include_rotation_tags and degrees:
                name = f"patch_{pid}_rot{degrees}.png"
            else:
                name = f"patch_{pid}.png"
            rotated.save(output_patches / name)
        rows.append({"patch_id": pid, "rotation_clockwise_degrees": degrees, "file": name})

    manifest_path = output_dir / "rotation_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("patch_id", "rotation_clockwise_degrees", "file"))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[done] Wrote rotated validation patches -> {output_patches}")
    print(f"[done] Rotation manifest -> {manifest_path}")
    print("[note] patch_0 was kept upright as the fixed top-left anchor.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create rotated validation patches")
    parser.add_argument("base_dir", nargs="?", default=".", help="Project directory containing patches/")
    parser.add_argument("--output", default="rotated_validation", help="Output directory")
    parser.add_argument("--seed", type=int, default=769, help="Random seed")
    parser.add_argument(
        "--mode",
        choices=("random", "random-nonzero", "cycle"),
        default="random",
        help="How to assign rotations to patches other than patch_0",
    )
    parser.add_argument(
        "--include-rotation-tags",
        action="store_true",
        help="Name files like patch_7_rot90.png. Leave off to mimic hidden tests with unlabelled rotations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    create_rotated_validation(
        patches_dir=base_dir / "patches",
        output_dir=base_dir / args.output,
        seed=args.seed,
        mode=args.mode,
        include_rotation_tags=args.include_rotation_tags,
    )


if __name__ == "__main__":
    main()