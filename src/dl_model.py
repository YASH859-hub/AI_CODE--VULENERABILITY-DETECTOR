"""Deep-learning model training for CodeBERT or sequence models."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a deep-learning vulnerability detector.")
    parser.add_argument("--train", type=Path, required=True, help="Path to the training split.")
    parser.add_argument("--validation", type=Path, required=True, help="Path to the validation split.")
    parser.add_argument("--model-out", type=Path, default=Path("models/dl_model.pt"), help="Output model path.")
    parser.add_argument("--architecture", choices=("codebert", "bilstm"), default="codebert", help="Model architecture.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(f"{args.architecture} training is not implemented yet.")


if __name__ == "__main__":
    main()
