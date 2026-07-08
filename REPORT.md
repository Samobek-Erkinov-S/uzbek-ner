# Uzbek NER: Fine-Tuning and Controlled Comparison on risqaliyevds/uzbek_ner

**Author:** Samobek Erkinov
**Task:** Named Entity Recognition fine-tuning (BERT/RoBERTa family) on the `risqaliyevds/uzbek_ner` dataset, 5 entity classes, maximizing quality.

---

## 1. Summary

I converted the entity-list annotations of `risqaliyevds/uzbek_ner` into a token-level BIO dataset (19,609 rows, 5 classes, 11 tags), trained a baseline, and ran a controlled comparison of four pretrained models under identical settings (same splits, seed, recipe; per-model tokenization audits). Results on the held-out data:

- **Final model: XLM-RoBERTa-large** — validation F1 **0.7246** on the five specified classes; **0.7139** on the untouched test split (agreement within 1.1 points of validation — model selection did not overfit). Token accuracy is reported alongside (baseline: 94.30% against an 86.5% all-O floor), but entity-level F1 is the primary metric — Section 5 explains why.
- On the **consistently-annotated core** (PERSON, ORG, and GPE+LOC merged; DATE excluded), the same model reaches **F1 0.8089**.
- The ~9-point gap between the headline and the core is **measured and attributed** to two documented annotation pathologies of the dataset: GPE/LOC double-labeling (−5.8 to −6.4 F1 across all four architectures) and DATE semantic noise (−2.0 to −2.2). These taxes are architecture-independent: no model choice recovers them.
- Model ranking (validation, 5-class): **XLM-R-large 0.7246 > XLM-R-base 0.7060 > mBERT 0.6976 > TahrirchiBERT 0.6816.** The monolingual-Uzbek hypothesis was refuted for this task; the deficit concentrates in foreign-name-dense classes, consistent with cross-lingual entity coverage — not Uzbek fluency — being the binding constraint for news NER.

Every number in this report traces to a specific notebook cell; the pipeline is fully reproducible from the frozen artifacts (Section 10).

## 2. Task and data

The dataset is 19,609 rows with two fields: `text` (a news passage, 87–999 characters) and `ner` (a dictionary mapping entity categories to lists of entity strings). **It is not token-annotated:** the annotations name which entities occur, not where. Building the token-level annotation layer is therefore the core engineering work of this task (Section 4), and its quality bounds everything downstream.

The dataset ships a single `train` split; train/validation/test splits were created here (Section 5).

## 3. Data analysis and label reconciliation (Phase 1)

Exploration surfaced ~29 category keys with heavy inconsistency, requiring reconciliation before any class selection:

- **Duplicate schemes:** `PER` and `PERSON` are the same concept under two names. A full-corpus check found **0 rows using both keys**, so they merge as a pure union. Same for `FAC`/`FACILITY`.
- **A dead and a decoy field:** `DATE` is populated directly (6,065 rows / 10,932 strings); `PERIOD` is a near-empty artifact (4 rows / 8 strings) and was discarded.
- **Noise keys:** `JCH-2022` (an entity value leaked into the key position), `RASUM`, `MISC` — excluded.
- **Junk values:** 200+ empty strings inside entity lists (DATE alone: 206); non-date fragments labeled DATE (e.g. `ikki kun` — a duration; `hali ma'lum`); junk in TIME (`—`, `90+2` — football match minutes).

**Class selection.** After reconciliation, the five most frequent categories — **GPE, LOC, ORG, PERSON, DATE** — together account for **88.3%** of all valid entity strings (153,007 of 173,262), lead the sixth-ranked category (MONEY) by a wide margin, and are exactly the PII-relevant types for depersonalization. Frequency table (rows containing the category / total entity strings): GPE 19,503 / 50,036 · LOC 18,898 / 38,751 · ORG 18,141 / 28,244 · PERSON 11,996 / 25,044 · DATE 6,065 / 10,932.

Based on the catalogued noise, two predictions were registered **before training**: DATE would be the weakest class (semantic label noise), and GPE↔LOC confusion would dominate the error profile. A third — PERSON recall depressed by surname-only mentions — was also registered; its outcome was instructive (Section 6).

## 4. The annotation layer: entity lists → BIO tags (Phase 2)

**Conversion rules** (implemented as a deterministic converter): entities are matched to whitespace-separated words after apostrophe normalization (six Unicode look-alikes canonicalized) and edge-punctuation stripping. Classes are processed in fixed priority order (PERSON, ORG, GPE, LOC, DATE); within a class, longer entities match first; a placed tag is never overwritten; every non-conflicting occurrence is tagged. All words of a multi-word entity must match exactly except the final word, which matches by prefix to tolerate Uzbek agglutinative suffixes (`Stokholm` accepts `Stokholmdagi`); the whole suffixed word joins the span. Strings shorter than 2 characters or longer than 6 words are treated as junk.

**Output:** 19,609 rows, 11 tags, O-share 86.5%, 99.2% of rows contain at least one span; **143,723 spans** total (GPE 50,108 · ORG 31,187 · PERSON 29,747 · LOC 21,670 · DATE 11,011). All rows pass automated BIO-validity and length checks (0 violations), re-verified after save/reload of the Arrow artifact.

**Match-rate audit.** Raw entity match rates (PERSON 97.8% · DATE 90.6% · ORG 81.7% · GPE 62.4% · LOC 42.4%) look alarming until decomposed. An absence audit (does the entity string occur in the stored text at all?) shows the low rates are a **data property, not a matching failure**: the stored texts are truncated at ~1,000 characters while the entity lists cover full articles, so context entities are frequently absent — the string `O'zbekiston` alone accounts for 81% of all GPE misses. Adjusted for absence, match rates are **PERSON 99.7 · DATE 97.0 · ORG 94.0 · GPE 91.6 · LOC 53.8**.

LOC's remaining deficit has its own mechanism: **7,993 rows (40.8% of the corpus) list identical strings under both GPE and LOC** (13,072 shared strings). The converter's priority rules resolve these ties toward GPE deterministically — producing zero incorrect tokens (blocked duplicates are entity-accounting events, not label errors) but registering as LOC misses. This same double-labeling was predicted to surface as GPE↔LOC confusion at evaluation time; it did (Section 6).

## 5. Experimental setup (Phases 3–5)

**Splits:** 80/10/10 with seed 42 → 15,687 / 1,961 / 1,961 rows. Per-split class balance verified (B-tag share of all tokens, %): train PERSON 1.73 · ORG 1.82 · GPE 2.93 · LOC 1.27 · DATE 0.65; validation 1.78 · 1.91 · 2.89 · 1.26 · 0.61; test 1.70 · 1.73 · 2.95 · 1.23 · 0.61 — every figure within 0.1 percentage points of the corpus-wide shares. The test split was touched exactly once, after all model selection concluded.

**Labels:** a frozen 11-tag inventory in fixed priority order (`O`, then B-/I- pairs for PERSON, ORG, GPE, LOC, DATE), shared by every run.

**Tokenization and max_length.** Word/token statistics: words per row min 10 · p50 89 · p99 128 · max 159; token lengths under the XLM-R tokenizer: min 28 · p50 196 · p90 253 · p99 289 · p99.5 303 · max 361 — a median of **2.2 subword pieces per Uzbek word**, quantifying how subword-hungry Uzbek is even for the most efficient of the three vocabularies tested (English typically runs ~1.2–1.4 on comparable vocabularies). Because vocabularies differ sharply in how they cut Uzbek, `max_length` was audited **per tokenizer** by counting training spans whose first subword would fall beyond each candidate budget:

| spans lost at | XLM-R (250k vocab) | TahrirchiBERT | mBERT |
|---|---|---|---|
| 256 | 653 | 1,695 | 7,722 |
| 320 | 23 | 162 | 712 |
| 384 | **0** | 5 | 13 |

The multilingual XLM-R vocabulary is the most token-efficient on Uzbek; mBERT the least by an order of magnitude — a shared budget of 256 would have silently deleted 6.7% of mBERT's training entities. (The baseline was trained at 256 before this audit was automated, costing 653 training spans, 0.57% — a documented footnote, not a repeated mistake: all comparison runs use their audited lengths.)

**Recipe (identical across all runs):** lr 2e-5, 4 epochs, effective batch 16, weight decay 0.01, 6% warmup, fp16, seed 42, best checkpoint selected by validation F1 each epoch.

**Metric — and why F1 is the headline rather than accuracy.** The assignment asks to maximize accuracy, so the choice of headline metric deserves an explicit justification. Token accuracy is structurally misleading for NER: the corpus is **86.5% O-tokens**, so a model that predicts *no entities at all* already scores 86.5% "accuracy" — and the trained models' ~94% token accuracy (baseline: 94.30%) sits only ~8 points above doing nothing, while their actual entity-finding quality differs enormously. The honest reading of the task's goal — a model that correctly finds entities — is measured by **entity-level precision/recall/F1** (seqeval), where a prediction counts only if the span's boundaries *and* type both match; this is the standard NER yardstick and the metric used for model selection here. Token accuracy is still reported for completeness, always anchored to its floor. Because the validation/test labels carry the same annotation noise as training, all scores measure **agreement with noisy references, not agreement with truth**.

## 6. Baseline results and decomposition (Phases 5–6)

**Baseline (XLM-R-base), validation:** F1 **0.7060** (P 0.675, R 0.740); token accuracy 0.9430 (vs the 86.5% all-O floor). Per class (F1): PERSON 0.885 · GPE 0.771 · ORG 0.673 · LOC 0.479 · DATE 0.450.

**Decomposition.** Three additional evaluation-side views isolate the annotation costs:

| view | micro-F1 | Δ | measures |
|---|---|---|---|
| 5 classes as specified | 0.7060 | — | the headline |
| GPE+LOC merged (eval only) | 0.7699 | +6.4 | the place double-labeling tax |
| DATE removed (eval only) | 0.7277 | +2.2 | the DATE semantic-noise tax |
| both (the core) | **0.7970** | +9.1 | ability on well-defined classes |

**Pre-registered prediction verdicts:**

1. **DATE weakest — confirmed.** F1 0.450, precision 0.399: gold's junk ("ikki kun", "oldinroq") taught an over-broad notion of DATE, and strict span matching double-punishes boundary disagreements.
2. **GPE↔LOC confusion, LOC suffering most — confirmed with its fingerprint.** LOC is the only class where recall (0.460) falls below precision — the signature of contested place-strings defaulting to GPE. Token-level confusion counts: LOC→GPE 730, GPE→LOC 441 (bidirectional — direct evidence the inconsistency is annotation-borne, not converter-induced), plus an unpredicted second axis: ORG↔GPE 643 combined, driven by place/institution metonymy (e.g. club names).
3. **PERSON surname-recall — refuted, instructively.** PERSON scored 0.885 with recall above precision. The surname-only convention (full name annotated once; later "Messi" left as O) is *shared* by training and evaluation labels, so the model's convention-following is invisible to the metric. Consistent annotation conventions cannot be detected by same-source evaluation; the real-world recall cost exists but is unmeasurable here.

Aggregate signature: precision below recall in every class except LOC — the **coverage tax**: the annotations capture only ~80–90% of true entities, so the model is regularly penalized for correctly tagging entities gold forgot.

## 7. Error analysis (validation gallery)

Manual inspection of disagreement rows confirms the mechanisms with named examples:

- **Annotation self-contradiction (ORG↔GPE):** one sentence labels «Tottenhem» ORG but «Brayton» GPE — two clubs in identical roles; the model tags both ORG, consistently, five times.
- **Coverage tax (model beats gold):** goalscorer "Knokert" untagged in gold, correctly tagged B-PERSON by the model; the Kurdistan Workers' Party ("Kurdiston ishchi partiyasi") untagged in gold across three mentions, correctly extracted as a 3-word ORG all three times. Scored as false positives.
- **Gold mislabels:** football players "Rayan" and "Myurrey" labeled **B-LOC** in gold (gazetteer collisions); the model declines.
- **Common-noun LOC:** gold tags generic administrative nouns ("mahalla", "viloyat", "tuman") as B-LOC; contradictory supervision makes the model waver — a third, distinct mechanism behind LOC's low recall.
- **Boundary fights:** gold "«Falmer»" vs model "«Falmer» stadioni"; DATE segmentations like "2023 yil" + "1 yanvardan" as two gold spans.

The majority of inspected "errors" are annotation defects rather than model defects — the qualitative counterpart of the +9.1-point core gap.

## 8. Controlled comparison (Phase 7)

Four pretrained checkpoints, one variable (the checkpoint), everything else held: same splits, seed, recipe; per-model tokenizer with audited max_length; identical scoring through all four evaluation views.

| run | max_len | P | R | F1 (5-class) | F1 place-merged | F1 no-DATE | F1 core | place-tax | DATE-drag |
|---|---|---|---|---|---|---|---|---|---|
| **XLM-R-large** | 384 | 0.704 | 0.746 | **0.7246** | 0.7825 | 0.7463 | **0.8089** | 0.0579 | 0.0217 |
| XLM-R-base | 256 | 0.675 | 0.740 | 0.7060 | 0.7699 | 0.7277 | 0.7970 | 0.0639 | 0.0217 |
| mBERT | 384 | 0.668 | 0.730 | 0.6976 | 0.7584 | 0.7177 | 0.7838 | 0.0608 | 0.0201 |
| TahrirchiBERT | 384 | 0.652 | 0.713 | 0.6816 | 0.7434 | 0.7015 | 0.7688 | 0.0618 | 0.0199 |

**Findings:**

1. **The taxes are architecture-independent.** Place-tax 5.8–6.4 points and DATE-drag 2.0–2.2 points across four architectures including a monolingual one; the per-class ranking (PERSON > GPE > ORG > LOC > DATE) is identical in every run. Label quality, not model choice, sets the class ordering and the ceilings of the noisy classes (DATE spans 0.450–0.462 across all four models).
2. **The monolingual hypothesis is refuted for this task.** TahrirchiBERT finishes last. Its deficit concentrates in foreign-name-dense classes (ORG −5.2, PERSON −3.4 vs baseline) while label-pinned classes are unmoved (DATE ±0.0) — consistent with cross-lingual coverage of transliterated names, not Uzbek fluency, being the binding constraint for news NER. Tokenization tells the same story: the monolingual vocabulary is more efficient than mBERT's but less than XLM-R's 250k-piece vocabulary (Section 5) — vocabulary size beats language fit.
3. **Capacity helps where labels allow.** XLM-R-large gains +1.9 overall, concentrated in ORG (+2.9), GPE (+1.4) and — a correction to the initial "pinned" reading — LOC (+4.3): LOC's deficit contains a capacity-responsive component alongside the irreducible label tax (large's place-tax, 0.0579, is correspondingly the smallest). Large's gains arrive mainly through precision (P +2.9 vs R +0.6), narrowing the coverage-tax gap from 6.5 to 4.2 points.

## 9. Final test result

The test split was evaluated exactly once, with the selected model (XLM-R-large), after the ledger was frozen.

**Test results (5 classes):** F1 **0.7139** (P 0.693, R 0.736) — within 1.1 points of the validation figure (0.7246), confirming that model selection did not overfit to validation. Per class:

| class | precision | recall | F1 | support |
|---|---|---|---|---|
| PERSON | 0.861 | 0.876 | 0.869 | 2,945 |
| GPE | 0.743 | 0.816 | 0.777 | 5,096 |
| ORG | 0.646 | 0.704 | 0.674 | 2,988 |
| LOC | 0.522 | 0.506 | 0.514 | 2,120 |
| DATE | 0.458 | 0.506 | 0.481 | 1,048 |

**Four evaluation views (test):** 5-class 0.7139 · place-merged 0.7751 · no-DATE 0.7329 · **core 0.7991**. The annotation taxes replicate on the held-out split — place-tax 6.1 points and DATE-drag 1.9 points, both consistent with the ranges measured across all four models on validation — and the class ranking (PERSON > GPE > ORG > LOC > DATE) and the precision-below-recall coverage signature (0.693 < 0.736) are likewise preserved. Per-class movements versus validation are within normal sampling variation for ~2,000-row splits (largest: ORG −2.8, DATE +2.1 points).

## 10. Limitations

- **Annotation coverage (~80–90%)** caps measurable precision: correct model tags on unannotated entities are scored as false positives; true recall against ground truth is unknowable from this data.
- **Label noise ceilings:** GPE/LOC double-labeling (40.8% of rows) and DATE semantic junk set architecture-independent ceilings, quantified above.
- **Convention blindness:** surname-only mentions train and evaluate as O; the metric cannot see this cost.
- **Heuristic matching:** the prefix rule can over-extend (e.g. hyphenated compounds, `Shvetsiya`→`Shvetsiyalik`); the ≤6-word filter excludes a few legitimate long official titles.
- **Comparability footnotes:** the baseline trained/evaluated at max_length 256 (653 training spans lost, 0.57%; validation gold reduced to 14,295 of ~14,420 spans), slightly flattering it relative to the 384-budget runs — the reported winner margin is therefore, if anything, understated.
- **Scope:** single-domain news text; single seed per configuration (seed 42); community models fine-tuned on this dataset exist on the Hugging Face Hub, but undocumented split differences prevent direct comparison.

## 11. Production recommendations (depersonalization use case)

1. **Merge GPE and LOC** for deployment: both are masked identically in depersonalization, and the merge recovers ~6 points of F1 at zero model cost (0.7825 with the current winner).
2. **Treat DATE separately:** either clean its annotations (highest-leverage data work) or handle dates with rules alongside the model; as-is it contributes noise disproportionate to its value.
3. **Model choice is a cost/quality dial:** XLM-R-large (+1.9 F1, ~2× inference cost) vs XLM-R-base; both run fully offline from local folders — no external calls, matching the on-premise requirement.
4. **Highest-leverage future work, in order:** annotation cleanup of GPE/LOC and DATE (worth ~9 points — more than any architecture change); **in-domain adaptation to legal/court text** (the training corpus is news; official legal register is out-of-distribution — confirmed by a controlled probe where an identical legal-style sentence detects the person in news order, 'Aziz Karimov', at score 0.80 but entirely misses the official surname-first form 'Karimov Aziz'); coreferent-mention expansion for PERSON; multi-seed runs for variance estimates; label-all-subtokens variant and CRF head as smaller refinements.

## 12. Reproducibility

One notebook per phase; every claim in this report traces to a cell. Frozen artifacts: the converted BIO dataset (Arrow), splits, label inventory (`label_list.json`), per-run ledger (`runs_ledger.csv`), conversion and frequency reports (CSV), final test report and views (txt/json), and the final model directory. Fixed seeds throughout (42); deterministic converter; per-model tokenization audits printed in-notebook. Repository layout and reproduction steps are in the README.
