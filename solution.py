"""
solution.py - Map reconstruction + VQA, fully offline after setup.

Usage:
    python solution.py [BASE_DIR]

BASE_DIR defaults to "." and must contain:
    test.csv
    patches/patch_0.png ... patch_N.png
    models/Qwen2-VL-7B-Instruct/   (downloaded by setup_model.py)
"""

from __future__ import annotations

import argparse
import os
import re
import time
import warnings
from pathlib import Path

import pandas as pd
from PIL import Image

from reconstruction_solver import (
    PATCH_NAME_RE,
    SolverConfig,
    build_compatibility,
    estimate_anchor_fit,
    estimate_overlap_width,
    grid_seam_energy,
    load_patches,
    reconstruct_grid,
    stitch_map,
)


warnings.filterwarnings("ignore")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

VALID_ANSWERS = {"1", "2", "3", "4", "5"}


def has_explicit_rotations(patches_dir: Path) -> bool:
    for patch_path in patches_dir.iterdir():
        if patch_path.suffix.lower() != ".png":
            continue
        match = PATCH_NAME_RE.match(patch_path.stem)
        if match and int(match.group(2) or 0):
            return True
    return False


def overlap_candidates_for_patch(patch_size: int) -> tuple[int, ...]:
    max_overlap = max(0, patch_size // 2)
    return tuple(sorted({0, *range(4, max_overlap + 1, 4)}))


def search_overlap_candidates(patches: dict[int, object]) -> tuple[int, ...]:
    patch_size = next(iter(patches.values())).shape[0]
    full_candidates = overlap_candidates_for_patch(patch_size)
    if len(full_candidates) <= 3:
        return full_candidates

    nonzero_candidates = tuple(width for width in full_candidates if width > 0)
    estimated = estimate_overlap_width(patches, candidates=nonzero_candidates, rotations=(0,))
    trimmed = {
        0,
        estimated,
        max(0, estimated - 8),
        max(0, estimated - 4),
        min(max(full_candidates), estimated + 4),
        min(max(full_candidates), estimated + 8),
    }
    return tuple(sorted(width for width in trimmed if width in set(full_candidates)))


def rotation_candidates_for_mode(selected_mode: str) -> list[tuple[str, tuple[int, ...]]]:
    if selected_mode == "none":
        return [("none", (0,))]
    if selected_mode == "all":
        return [("all", (0, 1, 2, 3))]
    return [("none", (0,)), ("all", (0, 1, 2, 3))]


def count_rotated_tiles(grid: list[list[tuple[int, int]]]) -> int:
    return sum(rot != 0 for row in grid for _, rot in row)


def candidate_score(
    patches: dict[int, object],
    grid_size: int,
    cfg: SolverConfig,
) -> tuple[float, float, int]:
    grid = reconstruct_grid(patches, grid_size, cfg)
    comp = build_compatibility(patches, cfg)
    seam_energy = grid_seam_energy(grid, comp)
    rotated_tiles = count_rotated_tiles(grid)
    score = seam_energy + 0.05 * rotated_tiles
    return score, seam_energy, rotated_tiles


def choose_rotation_mode(
    patches: dict[int, object],
    grid_size: int,
    selected_mode: str,
    has_rotation_tags: bool,
) -> tuple[str, tuple[int, ...], float]:
    if selected_mode != "auto":
        mode_name, rotations = rotation_candidates_for_mode(selected_mode)[0]
        rotation_penalty = 0.01 if has_rotation_tags and len(rotations) > 1 else 0.03
        if len(rotations) == 1:
            rotation_penalty = 0.0
        return mode_name, rotations, rotation_penalty

    full_rotations = (0, 1, 2, 3)
    no_score, overlap_width, _ = estimate_anchor_fit(patches, rotations=(0,), anchor_rotations=(0,))
    all_score, _, _ = estimate_anchor_fit(
        patches,
        rotations=full_rotations,
        overlap_width=overlap_width,
        anchor_rotations=(0,),
    )
    improvement = no_score - all_score
    threshold = max(0.002, 0.08 * no_score)
    print(
        f"[probe] upright_fit={no_score:.4f} | rotated_fit={all_score:.4f} | "
        f"gain={improvement:.4f} | threshold={threshold:.4f}"
    )

    if improvement > threshold:
        print("[probe] Selected rotation mode: all")
        return "all", full_rotations, 0.01 if has_rotation_tags else 0.03

    print("[probe] Selected rotation mode: none")
    return "none", (0,), 0.0


def choose_reconstruction_config(
    patches: dict[int, object],
    grid_size: int,
    selected_mode: str,
    patches_dir: Path,
) -> SolverConfig:
    overlap_candidates = search_overlap_candidates(patches)
    has_rotation_tags = has_explicit_rotations(patches_dir)
    mode_name, search_rotations, rotation_penalty = choose_rotation_mode(
        patches,
        grid_size,
        selected_mode,
        has_rotation_tags,
    )

    best_cfg = None
    best_score = float("inf")
    anchor_rotations = (0,)
    for overlap_width in overlap_candidates:
        candidate_cfg = SolverConfig(
            overlap_width=overlap_width,
            edge_width=1,
            gradient_weight=0.0,
            boundary_weight=0.8,
            beam_width=32,
            beam_candidates=8,
            refine_passes=0,
            refine_candidates=8,
            swap_refine_passes=0,
            block_refine_passes=0,
            rotations=search_rotations,
            anchor_rotations=anchor_rotations,
            rotation_penalty=rotation_penalty,
        )
        score, seam_energy, rotated_tiles = candidate_score(patches, grid_size, candidate_cfg)
        print(
            f"[search] mode={mode_name:4s} | overlap={overlap_width:2d} | "
            f"seam={seam_energy:.4f} | rotated={rotated_tiles:3d} | score={score:.4f}"
        )
        if score < best_score:
            best_score = score
            best_cfg = candidate_cfg

    assert best_cfg is not None
    print(
        f"[search] Selected overlap={best_cfg.overlap_width}px | "
        f"rotations={len(best_cfg.rotations)} states/piece | score={best_score:.4f}"
    )
    return SolverConfig(
        overlap_width=best_cfg.overlap_width,
        edge_width=1,
        gradient_weight=0.0,
        boundary_weight=0.8,
        beam_width=128,
        beam_candidates=24,
        refine_passes=2 if len(best_cfg.rotations) > 1 else 0,
        refine_candidates=12,
        swap_refine_passes=5,
        block_refine_passes=1,
        rotations=best_cfg.rotations,
        anchor_rotations=best_cfg.anchor_rotations,
        rotation_penalty=best_cfg.rotation_penalty,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map reconstruction + offline VQA")
    parser.add_argument("base_dir", nargs="?", default=".", help="Directory containing test.csv and patches/")
    parser.add_argument(
        "--rotation-mode",
        choices=("auto", "none", "all"),
        default="auto",
        help="Automatic by default. 'all' searches 0/90/180/270 rotations, 'none' disables rotation search.",
    )
    parser.add_argument("--allow-rotations", action="store_true", help="Deprecated alias for --rotation-mode all.")
    parser.add_argument("--reconstruct-only", action="store_true", help="Stop after writing reconstructed_map.png.")
    parser.add_argument("--fixed-config", action="store_true", help="Skip automatic config search.")
    parser.add_argument("--overlap-width", type=int, default=None, help="Known patch overlap width in pixels.")
    parser.add_argument("--beam-width", type=int, default=96, help="Beam width for --fixed-config.")
    parser.add_argument("--beam-candidates", type=int, default=16, help="Candidates per beam expansion.")
    parser.add_argument("--rotation-penalty", type=float, default=0.01, help="Penalty for non-zero rotations.")
    parser.add_argument("--refine-passes", type=int, default=2, help="Local rotation refinement passes.")
    parser.add_argument("--swap-refine-passes", type=int, default=4, help="Tile swap refinement passes.")
    parser.add_argument("--block-refine-passes", type=int, default=1, help="2x2 block refinement passes.")
    parser.add_argument(
        "--use-existing-map",
        action="store_true",
        help="Skip stitching and answer questions from existing reconstructed_map.png.",
    )
    parser.add_argument(
        "--model-size",
        choices=("2b", "7b"),
        default="7b",
        help="Qwen2-VL model size. 2b is much faster/lighter than 7b.",
    )
    parser.add_argument("--vqa-image-size", type=int, default=1280, help="Square image size sent to Qwen2-VL.")
    parser.add_argument(
        "--vqa-view-mode",
        choices=("full", "full-crops"),
        default="full-crops",
        help="Use only the full map or the full map plus zoomed crops.",
    )
    parser.add_argument("--vqa-max-new-tokens", type=int, default=8, help="Maximum answer tokens from Qwen2-VL.")
    parser.add_argument(
        "--vqa-prompt-style",
        choices=("strict", "reason-then-answer"),
        default="strict",
        help="Strict asks for only one digit. reason-then-answer allows brief reasoning.",
    )
    parser.add_argument("--questions-csv", default="test.csv", help="Question CSV filename inside base_dir.")
    parser.add_argument("--output-csv", default="submission.csv", help="Output answer CSV filename inside base_dir.")
    parser.add_argument("--limit-questions", type=int, default=None, help="Answer only the first N questions.")
    parser.add_argument("--question-id", default=None, help="Answer only one question id, for example practice_1.")
    return parser.parse_args()


def fixed_solver_config(patches: dict[int, object], selected_mode: str, args: argparse.Namespace) -> SolverConfig:
    rotations = (0, 1, 2, 3) if selected_mode == "all" else (0,)
    overlap_width = args.overlap_width
    if overlap_width is None:
        overlap_width = estimate_overlap_width(patches, rotations=rotations)
    rotation_penalty = args.rotation_penalty if len(rotations) > 1 else 0.0
    return SolverConfig(
        overlap_width=overlap_width,
        edge_width=1,
        gradient_weight=0.0,
        boundary_weight=0.8,
        beam_width=args.beam_width,
        beam_candidates=args.beam_candidates,
        refine_passes=args.refine_passes if len(rotations) > 1 else 0,
        refine_candidates=12,
        swap_refine_passes=args.swap_refine_passes,
        block_refine_passes=args.block_refine_passes,
        rotations=rotations,
        anchor_rotations=(0,),
        rotation_penalty=rotation_penalty,
    )


def load_model(model_dir: Path):
    try:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError("Missing VLM dependencies. Run: python -m pip install -r requirements.txt") from exc

    if not model_dir.exists():
        raise FileNotFoundError(f"Model not found at {model_dir}. Run: python setup_model.py")

    print(f"[model] Loading from {model_dir} ...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    print("[model] Loaded.")
    return model, processor, process_vision_info, torch


def build_map_views(map_img: Image.Image, image_size: int, view_mode: str) -> list[Image.Image]:
    full = map_img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    if view_mode == "full":
        return [full]

    width, height = map_img.size
    boxes = [
        (0, 0, width // 2, height // 2),
        (width // 2, 0, width, height // 2),
        (0, height // 2, width // 2, height),
        (width // 2, height // 2, width, height),
        (width // 4, height // 4, (3 * width) // 4, (3 * height) // 4),
    ]
    crops = [map_img.crop(box).resize((image_size, image_size), Image.Resampling.LANCZOS) for box in boxes]
    return [full, *crops]


def build_messages(
    question: str,
    options: list[str],
    map_img: Image.Image,
    image_size: int,
    view_mode: str,
    prompt_style: str,
) -> list[dict]:
    views = build_map_views(map_img, image_size=image_size, view_mode=view_mode)
    opts = "\n".join(f"{i + 1}. {option}" for i, option in enumerate(options))
    view_note = (
        "The first image is the full reconstructed map. The following images are zoomed map regions "
        "in this order: top-left, top-right, bottom-left, bottom-right, center. "
        if view_mode == "full-crops"
        else "The image is the full reconstructed map. "
    )
    task = (
        "Choose the option best supported by visible map evidence. Do not guess from general knowledge. "
        "If the map does not show enough evidence, choose 5. Options 1-4 are candidate answers; "
        "option 5 means not enough evidence."
    )
    answer_rule = (
        "Briefly check the visual evidence, then end with `Final: <digit>`."
        if prompt_style == "reason-then-answer"
        else "Reply with exactly one digit only: 1, 2, 3, 4, or 5."
    )
    prompt = f"{view_note}{task}\n\nQuestion: {question}\n{opts}\n\n{answer_rule}"
    content = [{"type": "image", "image": view} for view in views]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def query_model(model, processor, process_vision_info, torch, messages, max_new_tokens: int = 8) -> str:
    with torch.inference_mode():
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
        new_tokens = output_ids[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()


def normalize_answer(raw: str) -> int:
    text = (raw or "").strip()
    final_match = re.search(r"(?:final|answer|option)\s*[:\-]?\s*([1-5])\b", text, flags=re.IGNORECASE)
    if final_match:
        return int(final_match.group(1))

    exact_match = re.fullmatch(r"\s*([1-5])\s*[\.\)]?\s*", text)
    if exact_match:
        return int(exact_match.group(1))

    match = re.search(r"\b([1-5])\b", text)
    if not match:
        return 5
    answer = match.group(1)
    return int(answer) if answer in VALID_ANSWERS else 5


def answer_questions(
    model,
    processor,
    process_vision_info,
    torch,
    map_img: Image.Image,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    results = []
    total = len(test_df)
    for idx, (_, row) in enumerate(test_df.iterrows(), start=1):
        qid = row["id"]
        options = [row["option_1"], row["option_2"], row["option_3"], row["option_4"]]
        messages = build_messages(
            row["question"],
            options,
            map_img,
            image_size=args.vqa_image_size,
            view_mode=args.vqa_view_mode,
            prompt_style=args.vqa_prompt_style,
        )
        image_count = sum(1 for item in messages[0]["content"] if item["type"] == "image")
        print(
            f"[qa] Starting {idx}/{total} {qid} | images={image_count} | "
            f"size={args.vqa_image_size} | question='{row['question'][:70]}'",
            flush=True,
        )
        t0 = time.time()
        raw = query_model(
            model,
            processor,
            process_vision_info,
            torch,
            messages,
            max_new_tokens=args.vqa_max_new_tokens,
        )
        elapsed = time.time() - t0
        answer = normalize_answer(raw)
        print(f"[qa] Done {qid} in {elapsed:.1f}s | raw='{raw}' -> {answer}", flush=True)
        results.append({"id": qid, "question_num": qid, "option": answer})
    return pd.DataFrame(results)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    patches_dir = base_dir / "patches"
    test_csv = base_dir / args.questions_csv
    output_csv = base_dir / args.output_csv
    map_output = base_dir / "reconstructed_map.png"
    model_name = "Qwen2-VL-2B-Instruct" if args.model_size == "2b" else "Qwen2-VL-7B-Instruct"
    model_dir = base_dir / "models" / model_name

    print("=" * 60)
    print("MAP RECONSTRUCTION + VQA (OFFLINE)")
    print("=" * 60)
    print(f"Base dir: {base_dir.resolve()}")

    if args.use_existing_map:
        if not map_output.exists():
            raise FileNotFoundError(f"{map_output} not found. Run reconstruction first.")
        map_img = Image.open(map_output).convert("RGB")
        print(f"[stitch] Reusing existing map -> {map_output}")
    else:
        patches = load_patches(patches_dir)
        n = len(patches)
        grid_size = int(round(n**0.5))
        if grid_size * grid_size != n:
            raise ValueError(f"Patch count is not square: {n}")

        selected_mode = "all" if args.allow_rotations else args.rotation_mode

        if args.fixed_config:
            solver_cfg = fixed_solver_config(patches, selected_mode, args)
            print(
                f"[config] Fixed config | overlap={solver_cfg.overlap_width}px | "
                f"beam={solver_cfg.beam_width} | candidates={solver_cfg.beam_candidates} | "
                f"rotation_penalty={solver_cfg.rotation_penalty}"
            )
        else:
            solver_cfg = choose_reconstruction_config(patches, grid_size, selected_mode, patches_dir)

        grid = reconstruct_grid(patches, grid_size, solver_cfg)
        map_img = stitch_map(patches, grid, map_output, overlap_width=solver_cfg.overlap_width)

    if args.reconstruct_only:
        print("[done] Reconstruction-only mode; skipped VQA.")
        return

    model, processor, process_vision_info, torch = load_model(model_dir)
    test_df = pd.read_csv(test_csv)
    if args.question_id:
        test_df = test_df[test_df["id"] == args.question_id].copy()
        if test_df.empty:
            raise ValueError(f"No question with id {args.question_id} found in {test_csv}")
    if args.limit_questions is not None:
        test_df = test_df.head(args.limit_questions).copy()
    print(f"[qa] {len(test_df)} questions loaded")
    print(
        f"[qa] VQA config | model={args.model_size} | image_size={args.vqa_image_size} | views={args.vqa_view_mode} | "
        f"max_new_tokens={args.vqa_max_new_tokens} | prompt={args.vqa_prompt_style}"
    )

    submission_df = answer_questions(model, processor, process_vision_info, torch, map_img, test_df, args)
    submission_df.to_csv(output_csv, index=False)
    print(f"\n[done] Submission -> {output_csv}")
    print(submission_df.to_string(index=False))


if __name__ == "__main__":
    main()
