# Rui AI Projects — NLP Thesis-style Mini Research

## Goal
Build a reproducible NLP text classification project for R1 thesis-option MS applications:
- Baseline: TF-IDF + Logistic Regression
- Models: DistilBERT / BERT
- Add robustness + domain shift experiments
- Error analysis + short paper-style report

## Research Questions
1) How much do transformers outperform classic baselines in-domain?
2) Under domain shift (IMDb → SST-2), which model generalizes better?
3) Under noise (typos/word-drop/swap), which model is more robust?
4) What are the dominant error types? (negation, sarcasm, long context, rare entities, label noise)

## Setup (Colab)
```bash
pip install -r requirements.txt
