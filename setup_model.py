"""Download Qwen2-VL weights to ./models/ for offline use."""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download


MODEL_CHOICES = {
    "2b": ("Qwen/Qwen2-VL-2B-Instruct", "./models/Qwen2-VL-2B-Instruct"),
    "7b": ("Qwen/Qwen2-VL-7B-Instruct", "./models/Qwen2-VL-7B-Instruct"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Qwen2-VL model weights")
    parser.add_argument(
        "--model-size",
        choices=tuple(MODEL_CHOICES),
        default="7b",
        help="Use 2b for normal laptops; 7b needs much more GPU/RAM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_id, save_dir = MODEL_CHOICES[args.model_size]
    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading {model_id} -> {save_dir} ...")
    snapshot_download(
        repo_id=model_id,
        local_dir=save_dir,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*"],
    )
    print("Download complete. You can now run solution.py offline.")


if __name__ == "__main__":
    main()
