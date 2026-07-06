# ML-Based Source Code Vulnerability Detector

## Project Summary
A machine learning pipeline for detecting likely security vulnerabilities in C/C++ source code before deployment. The project combines engineered risk features, classical ML baselines, and deep-learning tokenization workflows to flag unsafe code patterns such as buffer overflows, dangerous API calls, and insecure control flow.

## What This Project Does
- Cleans and tokenizes source code from C/C++ functions
- Extracts engineered risk features like unsafe API calls, nesting depth, function length, and pointer usage
- Builds classical baselines such as XGBoost on engineered features
- Supports CodeBERT tokenization for transformer-based deep learning
- Adds explainability with SHAP to identify key vulnerability signals
- Evaluates models with precision, recall, and F1-score
- Includes a CLI scanner scaffold for practical code scanning

## Data Sources
### Primary: Devign
- Real-world labeled C/C++ functions from FFmpeg and QEMU
- Approximately 27,000 examples with vulnerability labels
- Based on actual CVEs and real bug patterns

### Optional: Juliet Test Suite
- NIST synthetic secure/insecure examples
- Useful for pipeline sanity checks and early validation

## Key Pipeline Components
### 1. Preprocessing
- Remove comments and normalize whitespace
- Clean code snippets into a `clean` column
- Extract risk features such as:
  - `func_length`
  - `num_tokens`
  - `num_pointers`
  - `risky_api_count`
  - `nesting_depth`
  - `num_conditions`
  - `num_returns`
  - presence flags for risky APIs: `strcpy`, `strcat`, `sprintf`, `vsprintf`, `gets`, `scanf`, `memcpy`
- Generate CountVectorizer token features
- Build CodeBERT tokenized encodings for deep learning

### 2. Classical ML Baseline
- Train XGBoost or similar tree-based models on engineered feature sets
- Compare baseline results to deep-learning models
- Save models under `models/`

### 3. Deep Learning
- Tokenize cleaned code with a CodeBERT tokenizer
- Prepare tensor inputs for models such as BiLSTM or fine-tuned CodeBERT
- Save encoded training and validation data for reproducible training

### 4. Explainability
- Use SHAP to explain vulnerability predictions
- Surface the code patterns behind flagged samples
- Improve trust and debugging for security review

### 5. CLI Scanner
- Provide a command-line interface for scanning `.c` files
- Load trained model artifacts and report likely vulnerable code
- Support future line-level risk highlighting and automation

## Repository Structure

```text
AI_CODE-VULNERABILTY-DETECTOR/
├── data/
│   ├── raw/                # Raw dataset files
│   └── processed/          # Cleaned CSVs, encoded tensors, vectorized features
├── models/                 # Saved model artifacts, tokenizer objects
├── notebooks/              # Exploratory analysis and experiments
├── results/                # Evaluation reports, plots, and diagnostics
├── src/
│   ├── preprocess.py       # Cleaning, feature extraction, tokenization
│   ├── classical_model.py  # Classical training scaffold
│   ├── dl_model.py         # CodeBERT tokenization/training helper
│   ├── evaluate.py         # Shared evaluation metrics
│   ├── explain.py          # SHAP explanation workflow
│   └── cli.py              # Scanner CLI entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── knowledge.md            # Implementation details and project summary
```

## Setup Instructions

### Windows
```powershell
py -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Usage Examples

### Preprocess data
```powershell
python -m src.preprocess --input data/processed/train.csv --output data/processed/train_clean.csv
```

### Generate token sequence features
```powershell
python -m src.preprocess --input data/processed/train.csv --output data/processed/train_clean.csv --vectorizer-output data/processed/train_vectors.csv
```

### Build CodeBERT tokenized encodings
```powershell
python -m src.preprocess --input data/processed/train.csv --output data/processed/train_clean.csv --encoded-output data/processed/train_enc.pt
```

### Prepare deep model input
```powershell
python -m src.dl_model --train data/processed/train.csv --validation data/processed/val.csv
```

### Run CLI scanner
```powershell
python -m src.cli path\to\file.c --model models/classical_model.pkl
```

## Evaluation Strategy
- Use a three-way split: train / validation / test
- Measure precision, recall, and F1
- Prioritize recall to avoid missing real vulnerabilities
- Compare classical and deep-learning models on the same validation splits

## Notes
- Keep raw data in `data/raw/` and generated datasets in `data/processed/`
- Save model artifacts in `models/`
- Save evaluation charts and reports in `results/`
- Use `knowledge.md` for the current implementation details and design rationale

## Git Workflow
```bash
git checkout -b feature/<short-description>
git add .
git commit -m "Update README and add knowledge documentation"
git push origin feature/<short-description>
```

## Future Improvements
- Add training and evaluation scripts for XGBoost and CodeBERT
- Implement line-level vulnerability localization in the CLI
- Add SHAP report generation and visualization
- Add automated tests for preprocessing and model components
- Improve dataset ingestion from Devign JSON and Juliet test format
