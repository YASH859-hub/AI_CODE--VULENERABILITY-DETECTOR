"""SHAP explanations for trained vulnerability detectors."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate SHAP explanations for model predictions.")
    parser.add_argument("--model", type=Path, required=True, help="Path to a saved model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to samples to explain.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/shap"), help="Directory for explanation outputs.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(f"SHAP explanations are not implemented yet for {args.model}.")


if __name__ == "__main__":
    main()
