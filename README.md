# AI Code Vulnerability Detector

FYP | ThreeCircuits

An applied AI/ML project for detecting vulnerable source-code snippets using a reproducible machine-learning pipeline. The repository is organized for dataset preparation, classical ML baselines, deep-learning experiments, evaluation, explainability, and a future command-line scanner.

## Project Objectives

- Build a vulnerability classification pipeline for source-code samples.
- Establish classical baselines using models such as Random Forest and XGBoost.
- Train deep-learning models such as CodeBERT and BiLSTM for code understanding.
- Evaluate models with consistent metrics, confusion matrices, and experiment outputs.
- Add SHAP-based explanations to make predictions easier to inspect.
- Package the final model behind a lightweight CLI scanner.

## Current Status

This repository contains the project structure, Python environment lockfile, starter source modules, notebook workspace, and local artifact folders. Model training and scanner internals are intentionally scaffolded so each stage can be implemented and reviewed cleanly.

## Repository Structure

```text
AI_CODE-VULNERABILTY-DETECTOR/
├── data/
│   ├── raw/                # original dataset files, for example devign.json
│   └── processed/          # cleaned CSV files and train/validation/test splits
├── notebooks/              # exploration, dataset analysis, and experiments
├── src/
│   ├── preprocess.py       # tokenisation, cleaning, and feature extraction
│   ├── classical_model.py  # Random Forest / XGBoost baseline training
│   ├── dl_model.py         # CodeBERT / BiLSTM training entry point
│   ├── evaluate.py         # shared evaluation metrics
│   ├── explain.py          # SHAP explanation workflow
│   └── cli.py              # command-line scanner entry point
├── models/                 # saved model artifacts
├── results/                # metrics, plots, reports, and confusion matrices
├── requirements.txt        # pinned Python dependencies
└── README.md
```

## Environment Setup

The environment was created with Python 3.10 on Windows. Use the virtual environment workflow below from the project root.

```powershell
py -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

For macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

The pinned environment includes CUDA 11.8 PyTorch wheels:

```text
torch==2.7.1+cu118
torchvision==0.22.1+cu118
```

If a teammate is using CPU-only hardware or a platform where the CUDA wheels are not suitable, install PyTorch from the official CPU wheel index first and regenerate `requirements.txt` for that machine.

## Core Dependencies

- Data and numerical computing: `numpy`, `pandas`, `scikit-learn`
- Classical ML: `xgboost`
- Deep learning: `torch`, `torchvision`
- Code/NLP modeling: `transformers`, `datasets`, `tokenizers`
- Evaluation and visualization: `matplotlib`, `seaborn`
- Explainability: `shap`
- Developer tooling: `tqdm`, `rich`, `jupyterlab`

## Data Workflow

Place original datasets in `data/raw/`. Generated datasets should be written to `data/processed/`, typically as:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
```

Large datasets, trained model files, and generated plots are ignored by Git by default. This keeps the repository lightweight and prevents accidental commits of heavy experiment artifacts. Use cloud storage, Git LFS, or a release artifact workflow if the team decides to version those files.

## Planned Pipeline

1. **Preprocessing**
   - Load the raw vulnerability dataset.
   - Normalize source-code text and labels.
   - Generate train/validation/test splits.
   - Extract baseline features or prepare transformer tokens.

2. **Classical Baselines**
   - Train Random Forest and XGBoost models.
   - Compare baseline performance against deep-learning models.
   - Save model artifacts under `models/`.

3. **Deep Learning**
   - Fine-tune CodeBERT for binary vulnerability classification.
   - Optionally train a BiLSTM baseline over token sequences.
   - Track validation metrics and save the best checkpoint.

4. **Evaluation**
   - Report accuracy, precision, recall, and F1-score.
   - Generate confusion matrices and class distribution plots.
   - Store metrics and charts in `results/`.

5. **Explainability**
   - Use SHAP to explain model decisions where practical.
   - Export explanation summaries for vulnerable and safe predictions.

6. **CLI Scanner**
   - Load the selected trained model.
   - Accept a file or directory path.
   - Return vulnerability predictions with confidence scores.

## Command Examples

The source modules currently expose CLI entry points and will be expanded as the implementation matures.

```powershell
python -m src.preprocess --input data/raw/devign.json --output-dir data/processed
python -m src.classical_model --train data/processed/train.csv --model xgboost
python -m src.dl_model --train data/processed/train.csv --validation data/processed/val.csv
python -m src.explain --model models/classical_model.pkl --data data/processed/test.csv
python -m src.cli path/to/source/file.py --model models/classical_model.pkl
```

## JupyterLab

Use notebooks for exploration and experiment analysis:

```powershell
jupyter lab
```

Keep notebooks focused on investigation. Production-ready logic should move into `src/` once it stabilizes.

## Reproducibility

- Use the pinned `requirements.txt` for consistent dependency versions.
- Keep raw, processed, and generated artifacts separated.
- Save model outputs under `models/` with descriptive filenames.
- Save plots, metrics, and reports under `results/`.
- Commit source-code changes separately from experiment artifacts.

## Git Workflow

Recommended team workflow:

```bash
git checkout -b feature/<short-description>
git add .
git commit -m "Add <meaningful change>"
git push origin feature/<short-description>
```

Open a pull request into `main` after testing the change locally.

## Roadmap

- Implement dataset preprocessing for Devign-style JSON input.
- Add baseline feature extraction and XGBoost training.
- Add CodeBERT fine-tuning with validation checkpointing.
- Add shared evaluation reports and confusion matrix export.
- Add SHAP explanations for classical and transformer models.
- Implement a production-ready CLI scanner.
- Add automated tests for preprocessing, metrics, and CLI behavior.

## License

License information has not been added yet. Add a license before public distribution.
