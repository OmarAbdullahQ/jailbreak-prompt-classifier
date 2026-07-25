# Prompt Classifier

This project builds and evaluates a classifier that detects whether a prompt is a jailbreak attempt or a benign request, optimizing models' thresholds and comparing traditional machine-learning models against an LLM (Claude Haiku 4.5) under several prompting strategies.

## Dataset

- Source: [`jackhhao/jailbreak-classification`](https://huggingface.co/datasets/jackhhao/jailbreak-classification) (Hugging Face)
- Two classes: `benign` and `jailbreak`

## Workflow

- **Exploratory Data Analysis** — duplicates, missing values, class balance, prompt length distributions, top words/bigrams
- **Preprocessing** — lowercasing, URL removal, lemmatization, stopword removal
- **Feature Engineering** — train/test split, TF-IDF vectorization, label encoding
- **Traditional ML Models** — Logistic Regression, Random Forest, SVM, Decision Tree; threshold tuning, classification reports, confusion matrices, ROC/PR curves, cross-validation
- **LLM Integration** — Claude Haiku 4.5 via AWS Bedrock, using Zero-shot, One-shot, Few-shot, and Reasoning prompting, evaluated on the same test split
- **Gradio Demo** — interactive comparison of SVM vs. Claude Haiku 4.5

## Key Findings

- SVM was the best-performing and most stable traditional ML model.
- The Decision Tree showed high variance between validation and test performance.
- The final comparison identifies the strongest prompting strategy against the ML baselines

## Team 5
