"""Deep-learning model entrypoint for CodeBERT tokenization and training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


DEFAULT_CODEBERT_MODEL = "microsoft/codebert-base"


class CodeBERTDataset(Dataset):
    def __init__(self, encodings: dict[str, Any], labels: torch.Tensor | None = None) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    df = pd.read_csv(path)
    if "func" not in df.columns and "clean" not in df.columns:
        raise ValueError("Input CSV must contain either 'func' or 'clean' text column.")
    if "target" not in df.columns:
        raise ValueError("Input CSV must contain a 'target' label column.")
    return df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tokenize code for CodeBERT deep learning training.")
    parser.add_argument("--train", type=Path, required=True, help="Path to the training CSV file.")
    parser.add_argument("--validation", type=Path, required=True, help="Path to the validation CSV file.")
    parser.add_argument("--model-out", type=Path, default=Path("models/dl_model.pt"), help="Path to save the fine-tuned model.")
    parser.add_argument("--tokenizer-out", type=Path, default=Path("models/codebert_tokenizer"), help="Directory to save the tokenizer.")
    parser.add_argument("--max-length", type=int, default=256, help="Maximum sequence length for CodeBERT inputs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--architecture", choices=("codebert", "bilstm"), default="codebert", help="Model architecture to prepare.")
    parser.add_argument("--tokenized-train-out", type=Path, default=Path("data/processed/train_tokenized.pt"), help="Path to save tokenized training features.")
    parser.add_argument("--tokenized-val-out", type=Path, default=Path("data/processed/val_tokenized.pt"), help="Path to save tokenized validation features.")
    return parser


def get_text_column(df: pd.DataFrame) -> pd.Series:
    if "clean" in df.columns:
        return df["clean"].astype(str)
    return df["func"].astype(str)


def build_codebert_tokenizer(model_name: str = DEFAULT_CODEBERT_MODEL) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataframe(df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> dict[str, Any]:
    texts = get_text_column(df).tolist()
    encoding = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "labels": torch.tensor(df["target"].astype(int).tolist(), dtype=torch.long),
    }


def save_tokenized_data(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)


def main() -> None:
    args = build_parser().parse_args()
    if args.architecture != "codebert":
        raise NotImplementedError("Only CodeBERT tokenization is supported by this entrypoint.")

    tokenizer = build_codebert_tokenizer()
    tokenizer.save_pretrained(args.tokenizer_out)

    train_df = load_dataframe(args.train)
    val_df = load_dataframe(args.validation)

    train_tokens = tokenize_dataframe(train_df, tokenizer, args.max_length)
    val_tokens = tokenize_dataframe(val_df, tokenizer, args.max_length)

    save_tokenized_data(train_tokens, args.tokenized_train_out)
    save_tokenized_data(val_tokens, args.tokenized_val_out)

    print(f"Saved CodeBERT tokenizer to: {args.tokenizer_out}")
    print(f"Saved tokenized train data to: {args.tokenized_train_out}")
    print(f"Saved tokenized validation data to: {args.tokenized_val_out}")
    print("CodeBERT tokenization completed.")


if __name__ == "__main__":
    main()
