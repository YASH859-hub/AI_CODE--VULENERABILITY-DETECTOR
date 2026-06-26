"""Command-line scanner for checking source files with a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan code for likely vulnerabilities.")
    parser.add_argument("path", type=Path, help="File or directory to scan.")
    parser.add_argument("--model", type=Path, default=Path("models/classical_model.pkl"), help="Path to a trained model.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.path.exists():
        raise FileNotFoundError(args.path)
    console.print(f"Scanner scaffold ready. Model loading is not implemented yet: {args.model}")


if __name__ == "__main__":
    main()
