# Map Reconstruction + VQA Offline Pipeline

This project reconstructs a shuffled map from `patches/patch_N.png`, then answers multiple-choice questions with a local VLM.

## What Changed

The old reconstruction used a one-pass greedy fill. That is fragile: one early bad patch placement makes every later cell worse. The new solver in `reconstruction_solver.py` uses a stronger jigsaw pipeline:

- Automatic overlap estimation from the anchored `patch_0`.
- Automatic normalization of filename-encoded patch rotations such as `patch_17_rot90.png`.
- Automatic fallback between upright-only reconstruction and full rotation search when filename metadata is absent.
- Direct overlap-region matching, not just boundary seam comparison.
- Vectorized pairwise seam compatibility for every candidate patch state.
- Beam search over the full 15x15 grid instead of committing to one greedy path.
- Exhaustive 2-opt style tile-swap refinement after the initial assembly.
- Optional local rotation/swap refinement for datasets that truly rotate patches.
- Overlap-aware stitching that averages shared pixels and writes the true mosaic size.
- `python solution.py` now chooses the reconstruction mode automatically. `--rotation-mode all` is still available as an override.
- Lazy VLM imports, so reconstruction can be tested without loading Torch/Qwen.

LoRA is not used for reconstruction because the reconstruction problem is geometry, not language adaptation. A LoRA adapter would only help the VQA stage if you have labeled map-question-answer examples. Without labeled data, it is more efficient and more reliable to improve the map assembly first.

## Setup

Run once in an internet-enabled environment:

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python setup_model.py
```

After setup, the notebook/script can run offline as long as the model weights are under:

```text
models/Qwen2-VL-7B-Instruct/
```

## Required Layout

```text
.
├── solution.ipynb
├── solution.py
├── reconstruction_solver.py
├── setup_model.py
├── requirements.txt
├── test.csv
├── patches/
│   ├── patch_0.png
│   ├── patch_1.png
│   └── ...
└── models/
    └── Qwen2-VL-7B-Instruct/
```

`patch_0` is treated as the fixed top-left anchor after any filename-based normalization has been applied.

## Run

Fast reconstruction-only check:

```bash
python solution.py --reconstruct-only
```

Full reconstruction + VQA:

```bash
python solution.py
```

Recommended VQA run after you know the reconstruction settings:

```bash
python solution.py . --rotation-mode all --fixed-config --overlap-width 32 --beam-width 64 --beam-candidates 12 --vqa-image-size 1280 --vqa-view-mode full-crops --vqa-max-new-tokens 8 --vqa-prompt-style strict
```

If GPU memory is low, use only the full map image:

```bash
python solution.py . --rotation-mode all --fixed-config --overlap-width 32 --beam-width 64 --beam-candidates 12 --vqa-image-size 1024 --vqa-view-mode full --vqa-max-new-tokens 8 --vqa-prompt-style strict
```

If strict one-digit answers are poor, allow a short explanation and extract the final digit:

```bash
python solution.py . --rotation-mode all --fixed-config --overlap-width 32 --beam-width 64 --beam-candidates 12 --vqa-image-size 1280 --vqa-view-mode full-crops --vqa-max-new-tokens 32 --vqa-prompt-style reason-then-answer
```

If you want to force full rotation search:

```bash
python solution.py --rotation-mode all
```

`--allow-rotations` is still accepted as a backwards-compatible alias.

## Rotated Patch Validation

The provided public patches are upright, but the hidden/test patches may be
rotated by 0/90/180/270 degrees. `patch_0` is kept upright and fixed as the
top-left anchor.

Create a local hidden-test-style validation set:

```bash
python make_rotated_validation.py . --output rotated_validation --seed 769 --mode random
```

This writes:

- `rotated_validation/patches/`
- `rotated_validation/rotation_manifest.csv`

Validate reconstruction on the rotated patches:

```bash
python solution.py rotated_validation --reconstruct-only --rotation-mode all --fixed-config --overlap-width 32 --beam-width 64 --beam-candidates 12 --refine-passes 1 --swap-refine-passes 2 --block-refine-passes 0
```

Run a small hyperparameter sweep:

```bash
python tune_reconstruction.py . --max-configs 4
```

For a larger sweep:

```bash
python tune_reconstruction.py . --full --max-configs 0
```

The tuner writes `tuning_outputs/tuning_results.csv` and
`tuning_outputs/best_reconstructed_map.png`. Reuse the best values with
`solution.py --fixed-config` flags.

Notebook execution:

```bash
jupyter nbconvert --to notebook --execute solution.ipynb --output solution_executed.ipynb --ExecutePreprocessor.timeout=3600
```

Outputs:

- `reconstructed_map.png` or `reconstructed_map_updated.png` if Windows has the original image locked.
- `submission.csv`

## Submission Format

```text
id,question_num,option
ques_1,ques_1,2
ques_2,ques_2,5
```

Options `1`-`4` are answers. Option `5` means skip.
