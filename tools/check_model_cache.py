"""tools/check_model_cache.py — verify HF cache presence for required models."""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def _model_dir_name(model_id: str) -> str:
    return f"models--{model_id.replace('/', '--')}"


def model_is_cached(model_id: str, *, cache_root: Path | None = None) -> bool:
    root = cache_root or Path(os.environ.get("HF_HUB_CACHE") or Path.home() / "hf_cache" / "hub")
    return (root / _model_dir_name(model_id)).exists()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Check whether a model exists in the local HF cache.")
    parser.add_argument("model_id")
    parser.add_argument("--cache-root", default=None)
    args = parser.parse_args(argv)
    cache_root = Path(args.cache_root) if args.cache_root else None
    if not model_is_cached(args.model_id, cache_root=cache_root):
        raise SystemExit(f"Model not found in cache: {args.model_id}")
    print(f"Model cached: {args.model_id}")


if __name__ == "__main__":
    main()
