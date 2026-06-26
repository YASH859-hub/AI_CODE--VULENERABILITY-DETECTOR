"""Dataset cleaning, tokenisation, and feature extraction."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess vulnerability datasets.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the raw dataset file.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"), help="Directory for processed outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(f"Preprocessing pipeline is not implemented yet for {args.input}.")


if __name__ == "__main__":
    main()
