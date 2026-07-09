# Uzbek NER — Fine-Tuning and Controlled Comparison

Token-level Named Entity Recognition on the `risqaliyevds/uzbek_ner` dataset
(classes: **PERSON, ORG, GPE, LOC, DATE**), built from entity-list annotations
via a deterministic BIO converter, with a four-model controlled comparison and a
measured decomposition of annotation noise.

**Final model: XLM-RoBERTa-large** — entity-level F1 **0.7246** (validation) /
**0.7139** (held-out test) on the five specified classes; **0.8089** on the
consistently-annotated classes (PERSON, ORG, and GPE+LOC merged, DATE excluded).

## What this project does

The dataset is **not** token-annotated — it lists which entities occur, not where.
The core engineering work was therefore building and verifying the token-level
annotation layer, then training and comparing models on it:

- **Data reconciliation** — merged duplicate label schemes (PER→PERSON), discarded
  noise keys and a near-empty decoy field; the five chosen classes cover 88.3% of
  all annotations.
- **BIO conversion** — a deterministic converter with apostrophe normalization,
  agglutinative-suffix matching, and priority-based conflict resolution;
- **Controlled comparison** — four pretrained models (XLM-R-large/base, mBERT,
  TahrirchiBERT) under identical splits, seed, and recipe, each with a per-tokenizer
  `max_length` audit for fairness.

Full analysis, every number, and all design decisions: **REPORT.md**.

## Repository layout

- `REPORT.md` — the complete write-up.
- `notebooks/` — one notebook per phase, in order, with outputs preserved:
  - `1_2_explore_and_convert.ipynb` — data forensics, label reconciliation, BIO conversion
  - `3_4_split_tokenize_align.ipynb` — 80/10/10 split, tokenization, label alignment
  - `5_6_baseline_and_analysis.ipynb` — baseline training, per-class analysis, error gallery
  - `7_comparison_and_test.ipynb` — four-model comparison, gated test, depersonalization demo
- `artifacts/` — `label_list.json`, `runs_ledger.csv`, `phase1_category_frequency.csv`,
  `phase2_conversion_report.csv`, `final_test_report.txt`, `final_test_views.json`.

## Reproducing

Open the notebooks in order on Google Colab (T4 GPU). Fixed seed 42 throughout.
Datasets and model weights are **not** committed (multi-GB); they regenerate
deterministically — the converted dataset from notebook 1–2, the models from
notebooks 5 and 7. Frozen copies live on Google Drive under `/ner`
(paths are set at the top of each notebook).

## Metric note

Results are reported as entity-level precision/recall/F1 (seqeval), not token
accuracy: the corpus is 86.5% non-entity tokens, so a model predicting nothing
scores 86.5% "accuracy" — F1 is the honest measure of entity-finding quality.
REPORT.md §5 explains this in full.
