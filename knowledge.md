# ML-Based Source Code Vulnerability Detector

## Project Overview
This repository implements a vulnerability detection pipeline for source code, focused on C/C++ functions. The goal is to predict whether a code snippet is likely to contain a security vulnerability before deployment, such as buffer overflows, injection flaws, or unsafe API usage.

The pipeline is designed to support both classical machine learning baselines and deep learning models, with explainability and a simple CLI scanner for practical use.

## Key Features
- Preprocess C/C++ code into cleaned token sequences
- Extract engineered risk features such as risky API calls, nesting depth, and pointer usage
- Train classical ML models (e.g. XGBoost) on engineered features
- Tokenize code for deep learning using CodeBERT-style embeddings
- Support SHAP explainability to show which code patterns trigger vulnerable predictions
- Evaluate using precision, recall, and F1-score, with recall prioritized for security use cases
- Provide a simple CLI scanner for scanning `.c` files and flagging risky code lines

## Datasets
### Primary Dataset: Devign
- Approximately 27,000 labeled C/C++ functions
- Derived from real-world projects such as FFmpeg and QEMU
- Labels are vulnerability-based and aligned with real CVEs
- Ideal for training realistic vulnerability detectors

### Optional Dataset: Juliet Test Suite
- NIST synthetic dataset of secure and insecure code snippets
- Useful for sanity-checking the pipeline and measuring robustness
- Good for early-stage validation before training on the primary dataset

## Pipeline Components
### A. Data Preprocessing
- Load raw or processed CSV splits from `data/processed/`
- Clean code by removing comments and normalizing whitespace
- Extract engineered features:
  - `func_length`
  - `num_tokens`
  - `num_pointers`
  - `risky_api_count`
  - `nesting_depth`
  - `num_conditions`
  - `num_returns`
  - presence flags for risky APIs like `strcpy`, `sprintf`, `gets`, `memcpy`, etc.
- Build token sequence features using `CountVectorizer`
- Tokenize code for CodeBERT models and store PyTorch encodings

### B. Classical ML Baseline
- Train a classical model using engineered features
- XGBoost is the preferred baseline for structured feature performance
- Evaluate on held-out validation and test splits
- Compare precision, recall, and F1

### C. Deep Learning Model
- Prepare token sequences for models such as BiLSTM
- Fine-tune CodeBERT embeddings for code vulnerability classification
- Save tokenized datasets and tokenizer artifacts
- Support model checkpointing and inference

### D. Explainability
- Apply SHAP explainability to model predictions
- Identify code constructs that contribute most strongly to vulnerability outputs
- Make predictions interpretable for security reviewers

### E. CLI Scanner
- Provide a lightweight command-line scanner in `src/cli.py`
- Accept a file or directory path for scanning
- Load a trained model and produce vulnerability warnings
- Support future extension to line-level risk highlighting

## Evaluation Strategy
- Use a 3-way split: train / validation / test
- Track precision, recall, F1, and accuracy
- Prioritize recall because false negatives are more dangerous in security
- Use confusion matrices and score reports to compare models

## Project Structure
- `src/preprocess.py` — code cleaning, risk feature extraction, CountVectorizer token features, CodeBERT tokenization
- `src/classical_model.py` — scaffold for classical ML model training
- `src/dl_model.py` — CodeBERT tokenization and deep-learning workflow
- `src/evaluate.py` — shared classification metrics
- `src/explain.py` — SHAP explanation workflow
- `src/cli.py` — scanner CLI entry point
- `data/processed/` — cleaned datasets and split CSV files
- `models/` — saved model artifacts and tokenizer/vocabulary files
- `results/` — evaluation outputs, plots, and reports
- `notebooks/` — exploratory analysis and experimentation notebooks

## Usage Examples
### Preprocess data
```powershell
python -m src.preprocess --input data/processed/train.csv --output data/processed/train_clean.csv
```

### Generate token sequence features
```powershell
python -m src.preprocess --input data/processed/train.csv --output data/processed/train_clean.csv --vectorizer-output data/processed/train_vectors.csv
```

### Tokenize for CodeBERT
```powershell
python -m src.preprocess --input data/processed/train.csv --output data/processed/train_clean.csv --encoded-output data/processed/train_enc.pt
```

### Prepare deep model data
```powershell
python -m src.dl_model --train data/processed/train.csv --validation data/processed/val.csv
```

### Scan code with CLI
```powershell
python -m src.cli path\to\file.c --model models/classical_model.pkl
```

## Notes and Best Practices
- Use `data/processed/` for cleaned and split datasets, not raw files
- Keep model artifacts under `models/` and results under `results/`
- Save tokenizer and encoded tensors for reproducible deep-learning training
- Validate with both Devign and optional synthetic data where possible
- Focus on recall when selecting the final model, since missing vulnerabilities is more costly than false positives

## Future Enhancements
- Add a full BiLSTM training pipeline on token sequences
- Implement heuristics for line-level risk highlighting in the CLI
- Add unit tests and data validation checks
- Scale the pipeline to larger codebases and additional languages
- Integrate model explainability reports into `results/`
