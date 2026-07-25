# Jailbreak Prompt Classifier — Classical ML vs. LLM Prompting

A binary text classifier that detects **jailbreak** prompts (attempts to bypass an AI system's safety guidelines) vs. **benign** prompts. Classical ML models (TF-IDF + Logistic Regression / Random Forest / SVM / Decision Tree) are compared against **Claude Haiku 4.5** (via AWS Bedrock) under four prompting strategies: zero-shot, one-shot, few-shot, and reasoning.

## Dataset

- **Source:** [`jackhhao/jailbreak-classification`](https://huggingface.co/datasets/jackhhao/jailbreak-classification) on Hugging Face.
- **Columns:** `prompt` (text), `type` (`jailbreak` / `benign`).
- **Size:** 1,044 rows — 527 `jailbreak` / 517 `benign` (near-balanced), 11 duplicates removed, 0 missing values, 0 conflicting labels.
- **Prompt length:** jailbreak prompts are much longer on average (mean ≈ 330 words) than benign ones (mean ≈ 86 words).
- 28 prompts contain URLs, 414 contain digits, 1,042 contain special characters.

## Methodology

1. **EDA** — checked shape, duplicates, missing values, label conflicts, word-count distributions per class, and top words/bigrams per class.
2. **Cleaning** — lowercase → remove URLs → normalize whitespace → lemmatize (spaCy `en_core_web_sm`) → remove stop words (keeping negations like `not`, `no`, `never`, `n't`, since they matter for jailbreak intent).
3. **Feature engineering** — `TfidfVectorizer(ngram_range=(1,2), min_df=2)`, fit on train only → 9,761 features.
4. **Split** — stratified 70% train / 15% validation / 15% test, `random_state=42`.
5. **Models trained** — Logistic Regression, Random Forest (300 trees), SVM (linear kernel), Decision Tree (`max_depth=20`).
6. **Threshold tuning** — swept thresholds 0.1–0.9 on validation probabilities, picked the F1-optimal threshold per model.
7. **Stability check** — 5-fold stratified CV on train+val, since the Decision Tree showed a suspicious validation/test discrepancy; also swept `max_depth` — neither showed a code issue, just typical decision-tree variance on a small dataset.
8. **LLM integration** — Claude Haiku 4.5 via AWS Bedrock (`boto3`), `temperature=0.0`, retried with exponential backoff. A strict system prompt asks for a single-word label (`jailbreak`/`benign`); a separate reasoning prompt asks for 2–3 sentences of reasoning followed by a parsed `Label: …` line.
   - **Zero-shot** — no examples.
   - **One-shot** — 1 labeled example (jailbreak).
   - **Few-shot** — 4 labeled examples (2 jailbreak, 2 benign).
   - **Reasoning** — free-form reasoning, then a final label.
   - Example shots are sampled from the training set only, keeping the test set unseen.
9. **Fair comparison** — all 4 ML models and all 4 LLM strategies are scored on the same held-out test set for an apples-to-apples comparison.
10. **Gradio demo** — interactive UI comparing the tuned SVM against Claude Haiku 4.5 across the four prompting strategies.

## Results

### Validation Set — Optimal Threshold per Model

| Model | Optimal Threshold | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM | 0.3 | 0.9740 | 0.9615 | 0.9677 |
| Logistic Regression | 0.4 | 0.9865 | 0.9359 | 0.9605 |
| Random Forest | 0.4 | 0.8706 | 0.9487 | 0.9080 |
| Decision Tree | 0.3 | 0.9041 | 0.8462 | 0.8742 |

### 5-Fold Cross-Validation (Stability Check)

| Model | F1 Mean | F1 Std |
|---|---|---|
| SVM | 0.9489 | 0.0180 |
| Logistic Regression | 0.9391 | 0.0206 |
| Random Forest | 0.9287 | 0.0187 |
| Decision Tree | 0.9050 | 0.0304 |

SVM has the best *and* most stable F1 across folds — the most reliable classical model.

### Final Comparison — ML Models vs. LLM Prompting Strategies (Test Set)

| Model / Strategy | Type | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| SVM | ML | 0.968 | 0.950 | 0.987 | 0.968 |
| Logistic Regression | ML | 0.968 | 0.962 | 0.974 | 0.968 |
| Decision Tree | ML | 0.974 | 0.962 | 0.987 | 0.974 |
| Random Forest | ML | 0.897 | 0.835 | 0.987 | 0.905 |
| Zero-Shot LLM | LLM | 0.974 | 0.952 | 1.00 | 0.975 |
| One-Shot LLM | LLM | 0.975 | 0.952 | 1.00 | 0.975 |
| Few-Shot LLM | LLM | 1.00 | 1.00 | 1.00 | 1.00 |
| Reasoning LLM | LLM | 0.975 | 0.952 | 1.00 | 0.975 |

**Best overall: Few-Shot LLM** (F1 = 1.00) — Claude Haiku 4.5 with 4 labeled examples classified the test set perfectly, edging out every classical model and the other prompting strategies.

## Key Findings

- **SVM is the strongest, most stable classical model** (highest CV mean F1, lowest variance); threshold tuning (0.3–0.4 instead of the default 0.5) improved F1 for every model except Decision Tree.
- **The Decision Tree's validation/test discrepancy is split noise, not a real effect** — confirmed via 5-fold CV and depth tuning.
- **Jailbreak prompts are structurally distinct** (much longer, distinct vocabulary), which is why even a simple TF-IDF + linear model performs well.
- **All four LLM prompting strategies matched or beat every classical model**, with **few-shot prompting reaching a perfect F1 of 1.00** — adding a small number of labeled examples gave Claude Haiku 4.5 a clear edge over zero-shot alone.

## Setup & Usage

```bash
git clone <this-repo-url>
cd <this-repo>
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Set AWS credentials for the Bedrock/Claude Haiku 4.5 calls (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`), then run `prompt_classifier.ipynb` top to bottom to reproduce the EDA, ML pipeline, LLM evaluation, and the Gradio demo.

## Tech Stack

`pandas`, `scikit-learn`, `spaCy`, `matplotlib`/`seaborn`, AWS Bedrock (`boto3`) — Claude Haiku 4.5, `gradio`, Hugging Face `datasets`.

## License

Add a license for this repository (e.g. MIT) here.
