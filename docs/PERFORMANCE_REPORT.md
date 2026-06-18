# BrandPulse AI — Performance Report (Phase 3)

## Comparison: Classical vs Deep Learning

| Model              |   accuracy |   macro_f1 |   weighted_f1 |
|:-------------------|-----------:|-----------:|--------------:|
| LogisticRegression |     0.7704 |     0.7233 |        0.7761 |
| ComplementNB       |     0.7734 |     0.6860 |        0.7603 |
| BiLSTM             |     0.7741 |     0.7153 |        0.7768 |

**Headline metric = Macro F1** (not accuracy), because the dataset is heavily
imbalanced (63% negative). Macro F1 weights all three classes equally, so it
rewards models that handle the rare *positive* / *neutral* classes — not just
the dominant *negative* one.

### Winner: **LogisticRegression** (highest macro F1 = 0.7233)

## Confusion Matrices
- Logistic Regression → `reports/cm_logreg.png`
- Bi-LSTM → `reports/cm_lstm.png`
- Side-by-side → `reports/confusion_comparison.png`

## Interpretation
- **Classical (TF-IDF + LogReg):** fast (trains in seconds), fully interpretable
  (we can read the driving words straight from coefficients), strong baseline.
- **Deep Learning (Bi-LSTM):** models word *order* and context ("not happy" ≠
  "happy"), at the cost of slower training and lower interpretability.
- **Our upgrades:** negation fusion (`not_good`), class weighting for imbalance,
  confidence-based "needs review" flagging, and coefficient explainability.

## Where models fail (the brief's question)
Both models most often confuse **neutral ↔ negative** — many complaints are phrased
flatly ("flight delayed 2 hrs") with no overtly negative words, and short neutral
questions can read as mild complaints. Sarcasm ("great, another delay 🙄") remains the
hardest case for the classical model; the LSTM does marginally better by using order.
