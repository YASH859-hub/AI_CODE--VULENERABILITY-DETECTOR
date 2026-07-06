import argparse
from pathlib import Path
import pickle
import re
from typing import Any

import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from transformers import AutoTokenizer

RISKY_APIS = [
    'strcpy',
    'strcat',
    'sprintf',
    'vsprintf',
    'gets',
    'scanf',
    'memcpy',
]

RISKY_API_PATTERN = re.compile(r'\b(' + '|'.join(map(re.escape, RISKY_APIS)) + r')\b')
TOKEN_PATTERN = re.compile(r'\w+|\S')
CONDITION_PATTERN = re.compile(r'\b(if|while|for)\b')
RETURN_PATTERN = re.compile(r'\breturn\b')


def clean_code(code_str: str) -> str:
    """Clean the input code string by removing comments and unnecessary whitespace."""
    code_str = re.sub(r'/\*.*?\*/', '', str(code_str), flags=re.DOTALL)
    code_str = re.sub(r'//.*', '', code_str)
    return re.sub(r'\s+', ' ', code_str).strip()


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    return pd.read_csv(path)


def extract_features(code_str: str) -> dict:
    """Extract numeric and risky API features from a code string."""
    clean_str = clean_code(code_str)
    lower_str = clean_str.lower()
    tokens = TOKEN_PATTERN.findall(clean_str)

    risky_counts = {api: 0 for api in RISKY_APIS}
    risky_api_count = 0
    for match in RISKY_API_PATTERN.finditer(lower_str):
        api_name = match.group(1)
        risky_counts[api_name] += 1
        risky_api_count += 1

    nesting = 0
    max_nesting = 0
    for char in clean_str:
        if char == '{':
            nesting += 1
            max_nesting = max(max_nesting, nesting)
        elif char == '}':
            nesting = max(nesting - 1, 0)

    result = {
        'func_length': len(clean_str),
        'num_tokens': len(tokens),
        'num_pointers': clean_str.count('*'),
        'risky_api_count': risky_api_count,
        'nesting_depth': max_nesting,
        'num_conditions': len(CONDITION_PATTERN.findall(lower_str)),
        'num_returns': len(RETURN_PATTERN.findall(lower_str)),
    }
    result.update({f'has_{api}': risky_counts[api] > 0 for api in RISKY_APIS})
    return result


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if 'func' not in df.columns:
        raise ValueError("The input DataFrame must contain a 'func' column.")
    df = df.copy()
    df['clean'] = df['func'].astype(str).apply(clean_code)
    feature_df = df['clean'].apply(extract_features).apply(pd.Series)
    return pd.concat([df, feature_df], axis=1)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def build_vectorizer(max_features: int = 5000, ngram_range: tuple[int, int] = (1, 2)) -> CountVectorizer:
    return CountVectorizer(
        token_pattern=r"\b\w+\b",
        lowercase=True,
        max_features=max_features,
        ngram_range=ngram_range,
        binary=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Preprocess code vulnerability datasets.')
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/processed/train.csv'),
        help='Path to the input CSV file containing code samples.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/processed/train_clean.csv'),
        help='Path to write the cleaned output CSV file.',
    )
    parser.add_argument(
        '--vectorizer-output',
        type=Path,
        default=Path('data/processed/train_vectors.csv'),
        help='Path to write token sequence feature CSV.',
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help='Max number of token features for CountVectorizer.',
    )
    parser.add_argument(
        '--codebert-model',
        type=str,
        default='microsoft/codebert-base',
        help='Hugging Face model name for CodeBERT tokenization.',
    )
    parser.add_argument(
        '--tokenizer-output',
        type=Path,
        default=Path('models/codebert_tokenizer'),
        help='Directory to save the CodeBERT tokenizer.',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=512,
        help='Maximum sequence length for CodeBERT inputs.',
    )
    parser.add_argument(
        '--encoded-output',
        type=Path,
        default=Path('data/processed/train_enc.pt'),
        help='Path to save the tokenized CodeBERT encoding.',
    )
    return parser


def build_token_sequence_features(df: pd.DataFrame, vectorizer: CountVectorizer) -> pd.DataFrame:
    if 'clean' not in df.columns:
        raise ValueError("DataFrame must contain a 'clean' column before building token sequences.")
    token_matrix = vectorizer.fit_transform(df['clean'].astype(str).tolist())
    feature_names = vectorizer.get_feature_names_out()
    token_df = pd.DataFrame(token_matrix.toarray(), columns=[f'token_{name}' for name in feature_names])
    token_df.index = df.index
    return token_df


def save_vectorizer(vect: CountVectorizer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(vect, f)


def build_codebert_tokenizer(model_name: str = "microsoft/codebert-base") -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_batch(df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 512, text_column: str = "clean") -> dict[str, Any]:
    if text_column not in df.columns:
        raise ValueError(f"DataFrame must include a '{text_column}' column for tokenization.")
    texts = df[text_column].astype(str).tolist()
    return tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt',
    )


def save_encoded_data(encoded: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(encoded, path)


def main() -> None:
    args = build_parser().parse_args()
    df = load_dataframe(args.input)
    cleaned_df = preprocess_dataframe(df)
    save_dataframe(cleaned_df, args.output)

    vectorizer = build_vectorizer(max_features=args.max_features)
    token_df = build_token_sequence_features(cleaned_df, vectorizer)
    save_dataframe(token_df, args.vectorizer_output)

    vocab_path = Path('models/vocab.pkl')
    save_vectorizer(vectorizer, vocab_path)

    tokenizer = build_codebert_tokenizer(args.codebert_model)
    encoded = tokenize_batch(cleaned_df, tokenizer, max_length=args.max_length)
    save_encoded_data(encoded, args.encoded_output)
    tokenizer.save_pretrained(args.tokenizer_output)

    print(f'Wrote cleaned dataset to: {args.output}')
    print(f'Wrote token sequence features to: {args.vectorizer_output}')
    print(f'Wrote CountVectorizer object to: {vocab_path}')
    print(f'Wrote tokenized CodeBERT encoding to: {args.encoded_output}')
    print(f'Wrote CodeBERT tokenizer to: {args.tokenizer_output}')


if __name__ == '__main__':
    main()
