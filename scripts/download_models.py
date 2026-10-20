"""Download the optional local models into the project data directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

MODELS = {
    "bge-embedding": "BAAI/bge-large-en-v1.5",
    "bge-reranker": "BAAI/bge-reranker-large",
    "judge": "Qwen/Qwen2.5-0.5B-Instruct",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="+",
        choices=tuple(MODELS),
        help="Models to download; use all three for the full GPU reproduction.",
    )
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    root = Path(__file__).resolve().parents[1]
    model_root = root / "data" / "models"
    model_root.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    for name in args.models:
        repo_id = MODELS[name]
        destination = model_root / repo_id.split("/")[-1]
        print(f"Downloading {repo_id} -> {destination}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=destination,
            max_workers=2,
        )
    print("Model download complete.")


if __name__ == "__main__":
    main()

