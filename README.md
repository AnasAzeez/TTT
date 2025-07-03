# Paper Evaluation Suite

This repository contains scripts and instructions to reproduce the experiments from our paper "". It includes two main pipelines:

1. **Batch True/False Prediction** using a causal language model (Meditron-70B).
2. **Comparative Model Evaluation** using Mistral’s chat API to judge pairs of responses.

---

## 📂 Repository Structure

```
├── Model_Pred.py       # Script for True/False predictions
├── Comparison.py       # Script for A vs. B model comparison
├── Dataset.zip         # Dataset used
└── README.md           # This file
```

---

## 🛠️ Setup and Installation

1. Clone this repository:

   ```bash
   git clone <repo_url>
   cd <repo_dir>
   ```

2. Create and activate a Python environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Set your API keys:

   ```bash
   export MISTRAL_API_KEY="<your_mistral_key>"
   ```

---

## 📖 Scripts Overview

### 1. Model\_prediction.py

**Purpose:** Run a pretrained causal LM to label questions as True or False.

**Key steps:**

- Load `epfl-llm/meditron-70b` via HuggingFace Transformers.
- Read CSV batches of questions from `Datset/batches/`.
- Prompt the model with a strict True/False instruction.
- Parse and save predictions to `outputs/pred_<batch_name>.csv`.

**Usage:**

```bash
python Model_Pred.py \
  --input_glob "Dataset/batches/batch_*.csv" \
  --model "epfl-llm/meditron-70b" \
  --output_dir "outputs/"
```

**Output:** CSV files named `pred_batch_*.csv` with a new `predicted_answer` column.

---

### 2. comparison.py

**Purpose:** Use Mistral’s chat API to compare two model responses across multiple medical criteria.

**Key steps:**

- Load comparison sheets (`batch_?_comparison.xlsx`) from `inputs/comparisons/`.
- For each row (question, Response A, Response B), call the `mistral-small-latest` model.
- Evaluate six criteria: correctness, helpfulness, harmfulness, reasoning, efficiency, bias.
- Parse JSON output with robust retry and backoff.
- Save expanded CSVs under `outputs/evaluated_<batch>_comparison.csv`.

**Usage:**

```bash
python comparison.py \
  --batches \
      inputs/comparisons/batch_1_comparison.xlsx \
      inputs/comparisons/batch_2_comparison.xlsx \
  --model "mistral-small-latest" \
  --api_key "$MISTRAL_API_KEY" \
  --output_dir "outputs/"
```

**Output:** CSVs including columns like `correctness.verdict`, `correctness.reason`, etc.

---

## 📈 Results and Analysis

After running both scripts, you can load the CSV outputs into pandas or your favorite analysis tool to reproduce the tables and figures in the paper:

```python
import pandas as pd
pred_df = pd.read_csv('outputs/pred_batch_1.csv')
comp_df = pd.read_csv('outputs/evaluated_batch_1_comparison.csv')
```

Use these DataFrames to compute accuracy, error rates, and to generate comparative bar charts and confusion matrices.

---

## ⚙️ Configuration Options

Both scripts accept command-line flags for:

- `--model` name (HuggingFace identifier)
- `--input_glob` or `--batches` list
- `--output_dir`
- `--max_tokens` or `--retries` (optional)

Run `python script.py --help` for full options.

---

## 📋 Dependencies

All required packages:

- `transformers` (for Meditron model)
- `torch`
- `mistralai` (for chat evaluation)
- `pandas`, `openpyxl` (for Excel)
- `tqdm`

---

## 🚀 Reproducing the Paper

1. Prepare your input CSVs/Excel files under `Dataset/`.
2. Run `Model_Pred.py` to generate T/F labels.
3. Run `comparison.py` to evaluate model pairs.
4. Use your analysis scripts (e.g., Jupyter notebooks) to recreate figures and tables.

Feel free to modify hyperparameters and model choices to explore variants.

---

## 📜 License

This work is released under the MIT License. See [LICENSE](LICENSE) for details.
