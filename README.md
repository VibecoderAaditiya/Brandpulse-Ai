# BrandPulse AI — Twitter Sentiment Analysis (NLP)

Classify airline tweets as **Positive / Neutral / Negative**, comparing a **Classical**
NLP pipeline (TF-IDF + Logistic Regression / Naive Bayes) against a **Deep Learning**
model (Embedding → Bi-LSTM), and serving both through a live **Streamlit dashboard**.

> Case study: *"BrandPulse AI"* — automated Voice-of-Customer analytics. Built as the
> NLP-engineer-intern project across 4 phases.

## Dataset
[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
— 14,640 human-labelled tweets, 3 classes, real Feb-2015 timestamps. Chosen over
Sentiment140 because it natively supports the required **3-class** problem (Sentiment140
is 2-class only).

## Project structure
```
brandpulse-ai/
├── data/
│   ├── raw/Tweets.csv              # original dataset
│   └── clean_tweets.csv            # cleaned output of Phase 1
├── src/preprocess.py               # shared cleaning pipeline (notebooks + app)
├── notebooks/
│   ├── 01_Preprocessing_and_Classical.ipynb   # Phase 1 + 2A
│   └── 02_Deep_Learning_LSTM.ipynb             # Phase 2B
├── models/                         # saved vectorizer, models, tokenizer
├── reports/                        # metrics, confusion matrices, wordclouds, report
├── dashboard/app.py                      # Phase 4 Streamlit dashboard
├── scripts/                        # reproducible build/train helpers
│   ├── build_nb01.py               #   regenerates notebook 01
│   ├── build_nb02.py               #   regenerates notebook 02
│   ├── train_lstm.py               #   standalone LSTM trainer (fast, logged)
│   ├── make_report.py              #   builds the Phase 3 comparison report
│   └── make_confusion_comparison.py
└── requirements.txt
```

## Setup
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; [nltk.download(p) for p in ['stopwords','wordnet','omw-1.4','punkt','punkt_tab','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
```

## Reproduce
```bash
# Phase 1 + 2A (cleaning + classical models)
jupyter nbconvert --to notebook --execute --inplace notebooks/01_Preprocessing_and_Classical.ipynb

# Phase 2B (LSTM — trains in ~20s on CPU after the import-order fix below)
jupyter nbconvert --to notebook --execute --inplace notebooks/02_Deep_Learning_LSTM.ipynb
# ...or train standalone with live per-epoch logging:
python scripts/train_lstm.py

# Phase 3 (comparison report)
python scripts/make_report.py && python scripts/make_confusion_comparison.py

# Phase 4 (dashboard)
streamlit run dashboard/app.py
```

## What we added beyond the brief
1. **Negation handling** — `"not good"` → `not_good` token, so bag-of-words models stop
   being fooled by negated sentiment. Directly addresses the brief's sarcasm question.
2. **Imbalance-aware evaluation** — class weighting + **macro-F1** as the headline metric
   instead of misleading accuracy on a 63%-negative dataset.
3. **Explainability** — the dashboard shows the top words that drove each classical
   prediction (from LogReg coefficients).
4. **Confidence buckets** — predictions below 60% confidence are flagged *"needs human
   review"* rather than forced into a label — realistic product behaviour.
5. **Real-timestamp trend line** — the 24h/period trend uses the dataset's actual
   timestamps, not simulated noise.
6. **Single shared preprocessing module** — identical cleaning at train and serve time
   (no train/serve skew).

## ⚠️ macOS arm64 gotcha (important)
On Apple Silicon, **import TensorFlow before pandas**. pandas 2.x loads pyarrow, whose
bundled `libarrow` ships its own `absl` thread-sync symbols; if loaded first they
interpose TF's threadpool mutex and `model.fit()` **deadlocks at 0% CPU forever**. Every
script that uses both imports `tensorflow` first (see top of `scripts/train_lstm.py`).

## Deliverables checklist
- [x] `01_Preprocessing_and_Classical.ipynb` (TF-IDF + LogReg + Naive Bayes)
- [x] `02_Deep_Learning_LSTM.ipynb` (Embedding + Bi-LSTM)
- [x] `dashboard/app.py` (Streamlit dashboard: predictor, pie chart, trend, live stream)
- [x] Performance report with comparison table + confusion matrices
- [x] `reports/BrandPulse_AI_Internship_Report.pdf` (full Persevex-branded report)
- [x] `reports/BrandPulse_AI_Project_Handbook.pdf` (plain-language guide + tutorials)
- [x] `reports/BrandPulse_AI_Presentation.pptx` (14-slide deck)
- [x] `reports/full_metrics.json` (real per-class precision/recall/F1 for all models)
- [x] `Live_Link.txt` (GitHub + Colab links placeholder)

## Document / asset generators
All three documents are regenerated from the real metrics in `reports/full_metrics.json`:
```bash
python scripts/make_full_metrics.py   # recompute per-class metrics from saved models
python scripts/make_doc_assets.py     # warm-theme figures (charts, confusion, preview)
python scripts/make_report_pdf.py     # the internship report PDF
python scripts/make_handbook_pdf.py   # the project handbook PDF
python scripts/make_ppt.py            # the presentation
```
