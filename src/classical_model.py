"""Classical machine-learning baselines for vulnerability detection."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a classical vulnerability detector.")
    parser.add_argument("--train", type=Path, required=True, help="Path to the training split.")
    parser.add_argument("--model-out", type=Path, default=Path("models/classical_model.pkl"), help="Output model path.")
    parser.add_argument("--model", choices=("random_forest", "xgboost"), default="xgboost", help="Model family to train.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(f"{args.model} training is not implemented yet.")


if __name__ == "__main__":
    main()
