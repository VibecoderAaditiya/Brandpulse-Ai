"""
BrandPulse AI — Sentiment Analytics Platform
============================================
A Streamlit application that loads the trained Classical (TF-IDF + Logistic
Regression) and Deep Learning (Embedding -> Bi-LSTM) models and turns raw tweets
into a Voice-of-Customer analytics product.

Design: a clean, executive-level SaaS analytics interface (Linear / Stripe /
Vercel / Apple HIG) — light, minimal, data-focused. The presentation layer is
fully self-contained; ALL model loading, prediction and analytics logic is
preserved unchanged from the backend.

Run from the project root:
    streamlit run dashboard/app.py

Navigation (sticky top bar):
    Predictor         — executive overview + live dual-model predictor
    Live Monitoring   — sentiment distribution, trend, live activity feed
    Model Comparison  — performance KPIs, comparison chart, table, confusion analysis
    Analytics         — word clouds, learning curves, diagnostics, dataset insights
"""

import os
import sys
import json
import pickle
import time

# Import TensorFlow BEFORE pandas/pyarrow to avoid the absl/libarrow threadpool
# deadlock on macOS arm64 (see README + scripts/train_lstm.py). Wrapped so the
# app still runs the classical-only views on machines without TF installed.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tensorflow as _tf  # noqa: F401
    _tf.config.threading.set_intra_op_parallelism_threads(4)
except Exception:
    _tf = None

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make src/ importable regardless of where streamlit is launched from
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.preprocess import clean_tweet, LABELS, ID2LABEL  # noqa: E402

import joblib  # noqa: E402

MODELS = os.path.join(ROOT, "models")
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports")

# ---------------------------------------------------------------------------
# Design tokens — clean SaaS palette (Apple HIG / Linear / Stripe)
# ---------------------------------------------------------------------------
BG = "#F5F5F7"
SURFACE = "#FFFFFF"
INK = "#1D1D1F"
INK_SOFT = "#6E6E73"
LINE = "rgba(0,0,0,0.08)"
ACCENT = "#007AFF"
SUCCESS = "#34C759"
WARNING = "#FF9F0A"
DANGER = "#FF453A"

SENT_COLORS = {"negative": DANGER, "neutral": INK_SOFT, "positive": SUCCESS}
CHART_SEQ = [ACCENT, SUCCESS, WARNING, "#5E5CE6", "#FF375F"]
CONFIDENCE_FLOOR = 0.60  # below this -> flag for human review (our upgrade)

st.set_page_config(page_title="BrandPulse AI", page_icon="◆",
                   layout="wide", initial_sidebar_state="collapsed")


# ===========================================================================
# THEME
# ===========================================================================
def inject_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg:#F5F5F7; --surface:#FFFFFF; --ink:#1D1D1F; --soft:#6E6E73;
            --line:rgba(0,0,0,0.08); --accent:#007AFF;
        }
        html, body, .stApp {
            background:#F5F5F7 !important;
            font-family:'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', sans-serif;
            color:#1D1D1F; -webkit-font-smoothing:antialiased;
        }
        .main .block-container { padding-top:0 !important; padding-bottom:4rem; max-width:1160px; }

        h1,h2,h3,h4 { color:#1D1D1F !important; letter-spacing:-.02em; font-family:'Inter',sans-serif; }
        p, label, .stMarkdown, span, li { color:#1D1D1F; font-family:'Inter',sans-serif; }
        .stCaption, [data-testid="stCaptionContainer"] { color:#6E6E73 !important; }
        a { color:#007AFF !important; text-decoration:none; }
        hr { border-color:rgba(0,0,0,0.06); }

        /* ---- Card primitive (subtle, no glass overload) ---- */
        .bp-card {
            background:#FFFFFF; border:1px solid rgba(0,0,0,0.08);
            border-radius:14px; padding:1.25rem 1.35rem; margin-bottom:1rem;
            box-shadow:0 1px 2px rgba(0,0,0,0.03);
        }
        .bp-card.flush { margin-bottom:0; height:100%; }

        /* Streamlit bordered container -> clean card */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background:#FFFFFF; border:1px solid rgba(0,0,0,0.08) !important;
            border-radius:14px; padding:1.2rem 1.3rem;
            box-shadow:0 1px 2px rgba(0,0,0,0.03);
        }

        /* ---- Sticky top bar: brand (left) + nav (right) ---- */
        div[data-testid="stHorizontalBlock"]:has(div[role="radiogroup"]) {
            position:sticky; top:0; z-index:1000;
            background:rgba(245,245,247,0.82);
            backdrop-filter:saturate(180%) blur(14px);
            -webkit-backdrop-filter:saturate(180%) blur(14px);
            border-bottom:1px solid rgba(0,0,0,0.08);
            padding:.55rem .25rem; margin-bottom:1.6rem; align-items:center;
        }
        div[role="radiogroup"] {
            gap:.15rem; justify-content:flex-end; flex-wrap:nowrap;
        }
        div[role="radiogroup"] label {
            border-radius:8px !important; padding:.4rem .85rem !important;
            margin:0 !important; cursor:pointer; transition:all .15s ease;
            color:#6E6E73 !important; font-weight:500 !important; font-size:.9rem;
        }
        div[role="radiogroup"] label:hover { color:#1D1D1F !important; background:rgba(0,0,0,0.035); }
        div[role="radiogroup"] label:has(input:checked) {
            background:rgba(0,122,255,0.10); color:#007AFF !important;
        }
        div[role="radiogroup"] label:has(input:checked) p { color:#007AFF !important; font-weight:600; }
        div[role="radiogroup"] [data-testid="stMarkdownContainer"] p { font-weight:500; margin:0; }
        div[role="radiogroup"] input { display:none; }

        /* ---- Buttons: Apple-blue, restrained ---- */
        .stButton > button {
            background:#007AFF; color:#fff; border:none;
            border-radius:10px; padding:.55rem 1.4rem; font-weight:600; font-size:.92rem;
            box-shadow:none; transition:background .15s ease, transform .1s ease;
        }
        .stButton > button:hover { background:#0a84ff; transform:translateY(-1px); }
        .stButton > button:active { transform:translateY(0); }
        .stButton > button[kind="secondary"] {
            background:#FFFFFF; color:#1D1D1F; font-weight:500;
            border:1px solid rgba(0,0,0,0.10);
        }
        .stButton > button[kind="secondary"]:hover {
            background:#FAFAFA; border-color:rgba(0,122,255,0.4); color:#007AFF;
        }

        /* ---- Inputs ---- */
        .stTextArea textarea, .stTextInput input,
        .stSelectbox div[data-baseweb="select"] > div {
            background:#FFFFFF !important; border:1px solid rgba(0,0,0,0.12) !important;
            border-radius:10px !important; color:#1D1D1F !important; font-size:.98rem !important;
        }
        .stTextArea textarea:focus, .stTextInput input:focus {
            border-color:#007AFF !important; box-shadow:0 0 0 3px rgba(0,122,255,0.12) !important;
        }

        /* ---- Dataframe ---- */
        div[data-testid="stDataFrame"] {
            background:#FFFFFF; border-radius:12px; border:1px solid rgba(0,0,0,0.08);
        }
        .js-plotly-plot { border-radius:10px; }
        div[data-testid="stImage"] img { border-radius:10px; }
        div[data-testid="stAlert"] { border-radius:10px; border:1px solid rgba(0,0,0,0.06); }
        [data-testid="stMetric"] { background:transparent; }

        /* hide default chrome */
        #MainMenu, footer, header[data-testid="stHeader"] { visibility:hidden; }
        [data-testid="stToolbar"] { display:none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def chart_style(fig, height=None, legend=True):
    """Consistent Plotly styling across the whole product."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK, family="Inter", size=12.5),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                    yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11)) if legend else dict(),
        showlegend=legend,
        margin=dict(t=24, l=6, r=6, b=6),
        colorway=CHART_SEQ,
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)", zeroline=False,
                     linecolor="rgba(0,0,0,0.10)", tickfont=dict(color=INK_SOFT, size=11))
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)", zeroline=False,
                     linecolor="rgba(0,0,0,0.10)", tickfont=dict(color=INK_SOFT, size=11))
    if height:
        fig.update_layout(height=height)
    return fig


# ===========================================================================
# CACHED LOADERS  (backend — unchanged)
# ===========================================================================
@st.cache_resource
def load_classical():
    vec = joblib.load(os.path.join(MODELS, "tfidf_vectorizer.pkl"))
    model = joblib.load(os.path.join(MODELS, "logreg_model.pkl"))
    return vec, model


@st.cache_resource
def load_lstm():
    keras_path = os.path.join(MODELS, "lstm_model.keras")
    if not os.path.exists(keras_path) or _tf is None:
        return None, None, None
    import tensorflow as tf
    model = tf.keras.models.load_model(keras_path)
    with open(os.path.join(MODELS, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    with open(os.path.join(MODELS, "lstm_config.json")) as f:
        cfg = json.load(f)
    return model, tokenizer, cfg


@st.cache_data
def load_data():
    path = os.path.join(DATA, "clean_tweets.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["tweet_created"] = pd.to_datetime(df["tweet_created"], utc=True, errors="coerce")
    return df.dropna(subset=["tweet_created"]).sort_values("tweet_created")


@st.cache_data
def load_metrics():
    """Aggregate metrics (fallback)."""
    out = {}
    for fn in ("classical_metrics.json", "lstm_metrics.json"):
        p = os.path.join(REPORTS, fn)
        if os.path.exists(p):
            out.update(json.load(open(p)))
    return out


@st.cache_data
def load_full_metrics():
    """Rich per-class metrics + confusion matrices (single source of truth)."""
    p = os.path.join(REPORTS, "full_metrics.json")
    if os.path.exists(p):
        return json.load(open(p))
    return None


# ===========================================================================
# PREDICTION HELPERS  (backend — unchanged)
# ===========================================================================
def predict_classical(text, vec, model):
    cleaned = clean_tweet(text)
    if not cleaned:
        return None
    X = vec.transform([cleaned])
    probs = model.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    return {"label": ID2LABEL[idx], "confidence": float(probs[idx]),
            "probs": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
            "cleaned": cleaned}


def predict_lstm(text, model, tokenizer, cfg):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    cleaned = clean_tweet(text)
    if not cleaned:
        return None
    seq = pad_sequences(tokenizer.texts_to_sequences([cleaned]),
                        maxlen=cfg["max_len"], padding="post", truncating="post")
    probs = model.predict(seq, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {"label": ID2LABEL[idx], "confidence": float(probs[idx]),
            "probs": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
            "cleaned": cleaned}


def top_contributing_words(text, vec, model, n=6):
    cleaned = clean_tweet(text)
    X = vec.transform([cleaned])
    idx = int(np.argmax(model.predict_proba(X)[0]))
    coefs = model.coef_[idx]
    feature_names = vec.get_feature_names_out()
    present = X.nonzero()[1]
    contrib = sorted(((feature_names[i], coefs[i] * X[0, i]) for i in present),
                     key=lambda t: t[1], reverse=True)
    return [w for w, c in contrib[:n] if c > 0]


# ===========================================================================
# UI PRIMITIVES
# ===========================================================================
def section_heading(title, sub=""):
    st.markdown(
        f"""
        <div style="margin:.4rem 0 1rem;">
          <div style="font-size:1.4rem;font-weight:700;color:{INK};letter-spacing:-.02em;">{title}</div>
          {f'<div style="color:{INK_SOFT};font-size:.95rem;margin-top:.2rem;">{sub}</div>' if sub else ''}
        </div>
        """, unsafe_allow_html=True)


def kpi_cards(items):
    """Compact KPI cards. items = list of (label, value, sub, color)."""
    cols = st.columns(len(items))
    for col, (label, value, sub, color) in zip(cols, items):
        col.markdown(
            f"""
            <div class="bp-card flush">
              <div style="font-size:.72rem;color:{INK_SOFT};font-weight:600;
                   text-transform:uppercase;letter-spacing:.04em;">{label}</div>
              <div style="font-size:1.55rem;font-weight:700;color:{color};
                   letter-spacing:-.02em;margin:.25rem 0 .1rem;
                   font-variant-numeric:tabular-nums;">{value}</div>
              <div style="font-size:.8rem;color:{INK_SOFT};">{sub}</div>
            </div>
            """, unsafe_allow_html=True)


def sentiment_pill(label, confidence):
    c = SENT_COLORS[label]
    return (f"<span style='display:inline-flex;align-items:center;gap:6px;"
            f"background:{c}14;color:{c};padding:4px 11px;border-radius:7px;"
            f"font-weight:600;font-size:.82rem;border:1px solid {c}2E;'>"
            f"<span style='width:7px;height:7px;border-radius:50%;background:{c};'></span>"
            f"{label.upper()} · {confidence:.0%}</span>")


def prob_distribution(probs):
    rows = ""
    for lbl in ["positive", "neutral", "negative"]:
        p = probs.get(lbl, 0.0)
        c = SENT_COLORS[lbl]
        rows += f"""
        <div style="margin:.5rem 0;">
          <div style="display:flex;justify-content:space-between;font-size:.82rem;margin-bottom:3px;">
            <span style="color:{INK};font-weight:500;text-transform:capitalize;">{lbl}</span>
            <span style="color:{INK_SOFT};font-variant-numeric:tabular-nums;">{p:.1%}</span></div>
          <div style="background:rgba(0,0,0,0.05);border-radius:6px;height:6px;overflow:hidden;">
            <div style="width:{p*100:.1f}%;height:100%;background:{c};border-radius:6px;"></div>
          </div>
        </div>"""
    return rows


def predictor_card(name, desc, result, explanation, signal_label, signal_html):
    """Twin result card — identical structure for both models."""
    if result is None:
        return (f'<div class="bp-card flush"><div style="font-weight:700;font-size:1.05rem;">'
                f'{name}</div><div style="color:{INK_SOFT};font-size:.82rem;">{desc}</div>'
                f'<div style="color:{INK_SOFT};margin-top:1rem;">Nothing left after cleaning — '
                f'try a longer sentence.</div></div>')
    review = result["confidence"] < CONFIDENCE_FLOOR
    flag = (f"<span style='margin-left:8px;background:{WARNING}1A;color:{WARNING};"
            f"padding:3px 9px;border-radius:6px;font-size:.72rem;font-weight:600;'>"
            f"⚠ needs review</span>") if review else ""
    return f"""
    <div class="bp-card flush">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <div style="font-weight:700;font-size:1.05rem;color:{INK};">{name}</div>
          <div style="color:{INK_SOFT};font-size:.8rem;">{desc}</div>
        </div>
      </div>
      <div style="margin:.95rem 0 .2rem;">{sentiment_pill(result['label'], result['confidence'])}{flag}</div>
      <div style="margin-top:.9rem;">{prob_distribution(result['probs'])}</div>
      <div style="margin-top:1rem;padding-top:.9rem;border-top:1px solid {LINE};">
        <div style="font-size:.72rem;color:{INK_SOFT};font-weight:600;text-transform:uppercase;
             letter-spacing:.04em;margin-bottom:.35rem;">Model explanation</div>
        <div style="font-size:.86rem;color:{INK};line-height:1.45;">{explanation}</div>
      </div>
      <div style="margin-top:.85rem;">
        <div style="font-size:.72rem;color:{INK_SOFT};font-weight:600;text-transform:uppercase;
             letter-spacing:.04em;margin-bottom:.4rem;">{signal_label}</div>
        {signal_html}
      </div>
      <div style="margin-top:.85rem;font-size:.78rem;color:{INK_SOFT};">
        Cleaned input&nbsp;·&nbsp;<span style="font-family:monospace;background:rgba(0,0,0,0.04);
        padding:2px 7px;border-radius:5px;color:{INK};">{result.get('cleaned') or '—'}</span>
      </div>
    </div>"""


def word_chips(words, color=ACCENT):
    if not words:
        return f"<span style='color:{INK_SOFT};font-size:.84rem;'>No strong positive keywords.</span>"
    return " ".join(
        f"<span style='background:{color}12;color:{color};padding:3px 10px;border-radius:7px;"
        f"font-size:.8rem;font-weight:500;margin:0 4px 4px 0;display:inline-block;'>{w}</span>"
        for w in words)


def confusion_heatmap(cm, title, base_color):
    """Native Plotly confusion-matrix heatmap (consistent cool theme)."""
    cm = np.array(cm, dtype=float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    labs = [l.capitalize() for l in LABELS]
    text = [[f"{int(cm[i,j])}<br>{norm[i,j]*100:.0f}%" for j in range(3)] for i in range(3)]
    scale = [[0, "#FFFFFF"], [1, base_color]]
    fig = go.Figure(go.Heatmap(
        z=norm, x=labs, y=labs, text=text, texttemplate="%{text}",
        textfont=dict(size=12, color=INK), colorscale=scale, showscale=False,
        zmin=0, zmax=1, xgap=3, ygap=3, hovertemplate="%{y} → %{x}<extra></extra>"))
    fig.update_layout(title=dict(text=title, font=dict(size=13, color=INK)),
                      xaxis_title="Predicted", yaxis_title="Actual")
    fig.update_yaxes(autorange="reversed")
    return chart_style(fig, height=320, legend=False)


# ===========================================================================
# APP BOOTSTRAP
# ===========================================================================
inject_theme()

vec, clf = load_classical()
lstm_model, tokenizer, lstm_cfg = load_lstm()
df = load_data()
FM = load_full_metrics()
agg = load_metrics()

# Build a tidy metrics dict that works whether or not full_metrics exists
PRETTY = {"LogisticRegression": "Logistic Regression",
          "ComplementNB": "Complement Naive Bayes", "BiLSTM": "Bi-LSTM"}


def macro_precision_recall(model_block):
    pcs = model_block["per_class"]
    prec = np.mean([pcs[c]["precision"] for c in LABELS])
    rec = np.mean([pcs[c]["recall"] for c in LABELS])
    return prec, rec


# ---- Sticky top bar: brand (left) + nav (right) ----
bar_l, bar_r = st.columns([1, 2.1])
with bar_l:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:9px;padding-left:.35rem;'>"
        f"<span style='width:22px;height:22px;border-radius:6px;background:{ACCENT};"
        f"display:inline-flex;align-items:center;justify-content:center;color:#fff;"
        f"font-weight:800;font-size:.8rem;'>B</span>"
        f"<span style='font-weight:700;font-size:1.02rem;letter-spacing:-.01em;color:{INK};'>"
        f"BrandPulse AI</span></div>", unsafe_allow_html=True)
with bar_r:
    page = st.radio("nav", ["Predictor", "Live Monitoring", "Model Comparison", "Analytics"],
                    horizontal=True, label_visibility="collapsed")

models_ready = lstm_model is not None


# ===========================================================================
# PAGE 1 — PREDICTOR  (Executive Overview + Live Predictor)
# ===========================================================================
if page == "Predictor":
    # ---- Section 1: Executive Overview ----
    st.markdown(
        f"""
        <div style="margin:.2rem 0 1.1rem;">
          <div style="font-size:2.1rem;font-weight:800;letter-spacing:-.03em;color:{INK};
               line-height:1.1;">BrandPulse AI</div>
          <div style="font-size:1.15rem;font-weight:600;color:{INK};margin-top:.15rem;">
            Voice of Customer Intelligence</div>
          <div style="color:{INK_SOFT};font-size:.98rem;margin-top:.4rem;max-width:680px;">
            Compare Classical NLP and Deep Learning models while monitoring customer
            sentiment in real time.</div>
        </div>
        """, unsafe_allow_html=True)

    if FM:
        best = max(FM["models"], key=lambda m: FM["models"][m]["macro_f1"])
        b = FM["models"][best]
        ds = FM["test_size"]
        kpi_cards([
            ("Accuracy", f"{b['accuracy']*100:.1f}%", PRETTY[best], INK),
            ("Macro F1", f"{b['macro_f1']:.3f}", "all classes equal", SUCCESS),
            ("Test Set", f"{ds:,}", "held-out tweets", ACCENT),
            ("Best Model", "Logistic Reg.", "vs Bi-LSTM & NB", INK),
        ])
    elif agg:
        best = max(agg, key=lambda m: agg[m]["macro_f1"])
        kpi_cards([
            ("Accuracy", f"{agg[best]['accuracy']*100:.1f}%", PRETTY.get(best, best), INK),
            ("Macro F1", f"{agg[best]['macro_f1']:.3f}", "all classes equal", SUCCESS),
            ("Models", "3", "trained & compared", ACCENT),
            ("Best Model", "Logistic Reg.", "headline metric", INK),
        ])

    st.markdown("<div style='height:.6rem;'></div>", unsafe_allow_html=True)

    # ---- Section 2: Live Sentiment Predictor ----
    section_heading("Live Sentiment Predictor",
                    "Enter any message to compare both models side by side.")

    if "tweet_text" not in st.session_state:
        st.session_state.tweet_text = "I waited 4 hours just to get a cold burger."

    examples = {
        "Amazing Service": "The service was amazing, the crew went above and beyond!",
        "Lost Luggage": "@united my bag is lost again, this is completely unacceptable",
        "Flight Delay": "Flight delayed 3 hours with no explanation, not happy at all",
        "Product Question": "What time does boarding start for flight 482?",
    }
    ecols = st.columns(len(examples))
    for col, (tag, ex) in zip(ecols, examples.items()):
        if col.button(tag, key=f"ex_{tag}", use_container_width=True, type="secondary"):
            st.session_state.tweet_text = ex

    text = st.text_area("Tweet text", key="tweet_text", height=96,
                        label_visibility="collapsed",
                        placeholder="e.g. The crew went above and beyond on my flight today…")
    analyze = st.button("Analyze sentiment", type="primary")

    if not models_ready:
        st.info("Deep-Learning model not loaded — showing the Classical verdict only. "
                "Train notebook 02 (or install TensorFlow) to enable the Bi-LSTM.")

    if analyze or text.strip():
        r = predict_classical(text, vec, clf)
        words = top_contributing_words(text, vec, clf) if r else None
        c_expl = ("Linear model over TF-IDF features — the verdict is the weighted sum of "
                  "the keywords below, so every decision is fully traceable.")
        c_signal = word_chips(words, ACCENT)

        r2 = predict_lstm(text, lstm_model, tokenizer, lstm_cfg) if models_ready else None
        d_expl = ("Sequence model — reads word order in both directions, so it captures "
                  "context like negation ('not happy' ≠ 'happy') rather than isolated words.")
        if r2:
            tok_count = len((r2.get("cleaned") or "").split())
            d_signal = (f"<span style='color:{INK_SOFT};font-size:.84rem;'>Context-aware over "
                        f"<b style='color:{INK};'>{tok_count}</b> cleaned tokens · "
                        f"40-token sequence window.</span>")
        else:
            d_signal = f"<span style='color:{INK_SOFT};font-size:.84rem;'>Model not loaded.</span>"

        col1, col2 = st.columns(2)
        col1.markdown(
            predictor_card("Classical NLP", "TF-IDF + Logistic Regression", r,
                           c_expl, "Important words", c_signal), unsafe_allow_html=True)
        col2.markdown(
            predictor_card("Deep Learning", "Embedding + Bi-LSTM", r2,
                           d_expl, "How it read this", d_signal), unsafe_allow_html=True)


# ===========================================================================
# PAGE 2 — LIVE MONITORING
# ===========================================================================
elif page == "Live Monitoring":
    section_heading("Live Monitoring",
                    "Real-time sentiment distribution and movement across the dataset.")
    if df is None:
        st.error("data/clean_tweets.csv not found — run notebook 01 first.")
        st.stop()

    dist = df["airline_sentiment"].value_counts()
    total = int(dist.sum())
    kpi_cards([
        ("Total Tweets", f"{total:,}", "in dataset", INK),
        ("Negative", f"{dist.get('negative',0)/total:.0%}", f"{dist.get('negative',0):,}", DANGER),
        ("Neutral", f"{dist.get('neutral',0)/total:.0%}", f"{dist.get('neutral',0):,}", INK_SOFT),
        ("Positive", f"{dist.get('positive',0)/total:.0%}", f"{dist.get('positive',0):,}", SUCCESS),
    ])
    st.markdown("<div style='height:.8rem;'></div>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Sentiment Distribution**")
            d = dist.reindex(LABELS).fillna(0)
            fig = px.pie(values=d.values, names=d.index, hole=0.62,
                         color=d.index, color_discrete_map=SENT_COLORS)
            fig.update_traces(textinfo="percent",
                              marker=dict(line=dict(color="#FFFFFF", width=2)))
            st.plotly_chart(chart_style(fig, height=310), use_container_width=True)
    with right:
        with st.container(border=True):
            st.markdown("**Sentiment Trend** &nbsp;<span style='color:#6E6E73;font-size:.82rem;'>"
                        "real timestamps · 6h buckets</span>", unsafe_allow_html=True)
            ts = (df.set_index("tweet_created")
                    .groupby([pd.Grouper(freq="6h"), "airline_sentiment"])
                    .size().reset_index(name="count"))
            fig2 = px.line(ts, x="tweet_created", y="count", color="airline_sentiment",
                           color_discrete_map=SENT_COLORS, markers=True)
            fig2.update_traces(line=dict(width=2.4))
            st.plotly_chart(chart_style(fig2, height=310), use_container_width=True)

    st.markdown("<div style='height:.4rem;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Live Activity Feed**")
        fc1, fc2, fc3 = st.columns([2, 1, 1])
        with fc1:
            airline = st.selectbox(
                "Filter airline",
                (["All"] + sorted(df["airline"].dropna().unique()))
                if "airline" in df.columns else ["All"], label_visibility="collapsed")
        with fc2:
            speed = st.slider("Per refresh", 1, 10, 3, label_visibility="collapsed")
        with fc3:
            go_stream = st.button("Stream 30", type="primary", use_container_width=True)
        if go_stream:
            feed = df if airline == "All" or "airline" not in df.columns \
                else df[df["airline"] == airline]
            feed = feed.sample(min(30, len(feed)), random_state=1)
            placeholder = st.empty()
            rolling = {"negative": 0, "neutral": 0, "positive": 0}
            for i in range(0, len(feed), speed):
                batch = feed.iloc[i:i + speed]
                with placeholder.container():
                    for _, row in batch.iterrows():
                        s = row["airline_sentiment"]; rolling[s] += 1; c = SENT_COLORS[s]
                        st.markdown(
                            f"<div style='display:flex;gap:11px;align-items:flex-start;"
                            f"padding:.55rem .1rem;border-bottom:1px solid {LINE};'>"
                            f"<span style='flex:0 0 auto;width:8px;height:8px;border-radius:50%;"
                            f"background:{c};margin-top:6px;'></span>"
                            f"<span style='color:{INK};font-size:.9rem;line-height:1.4;'>"
                            f"{row['text'][:140]}</span></div>", unsafe_allow_html=True)
                    st.caption(f"Running tally — negative {rolling['negative']} · "
                               f"neutral {rolling['neutral']} · positive {rolling['positive']}")
                time.sleep(0.4)


# ===========================================================================
# PAGE 3 — MODEL COMPARISON  (+ Confusion Matrix Analysis)
# ===========================================================================
elif page == "Model Comparison":
    section_heading("Model Comparison",
                    "Performance on the held-out test set. Headline metric is macro-F1 — "
                    "the data is 63% negative, so accuracy alone misleads.")

    source = FM["models"] if FM else agg
    if not source:
        st.info("Run the notebooks / scripts/make_full_metrics.py to generate metrics.")
        st.stop()

    order = [m for m in ["LogisticRegression", "ComplementNB", "BiLSTM"] if m in source]
    best = max(source, key=lambda m: source[m]["macro_f1"])
    b = source[best]

    # ---- performance summary cards: Accuracy, Precision, Recall, Macro F1, Weighted F1 ----
    if FM:
        prec, rec = macro_precision_recall(b)
    else:
        prec, rec = float("nan"), float("nan")
    summary = [("Accuracy", f"{b['accuracy']*100:.1f}%", INK),
               ("Precision", f"{prec:.3f}" if prec == prec else "—", ACCENT),
               ("Recall", f"{rec:.3f}" if rec == rec else "—", ACCENT),
               ("Macro F1", f"{b['macro_f1']:.3f}", SUCCESS),
               ("Weighted F1", f"{b['weighted_f1']:.3f}", WARNING)]
    cols = st.columns(5)
    for col, (lab, val, c) in zip(cols, summary):
        col.markdown(
            f"<div class='bp-card flush' style='text-align:left;'>"
            f"<div style='font-size:.7rem;color:{INK_SOFT};font-weight:600;text-transform:uppercase;"
            f"letter-spacing:.04em;'>{lab}</div>"
            f"<div style='font-size:1.35rem;font-weight:700;color:{c};margin-top:.2rem;"
            f"font-variant-numeric:tabular-nums;'>{val}</div></div>", unsafe_allow_html=True)
    st.caption(f"Best model: **{PRETTY[best]}** · metrics shown for the winner.")
    st.markdown("<div style='height:.6rem;'></div>", unsafe_allow_html=True)

    cL, cR = st.columns([1.15, 1])
    with cL:
        with st.container(border=True):
            st.markdown("**Metric Comparison**")
            fig = go.Figure()
            palette = {"accuracy": ACCENT, "macro_f1": SUCCESS, "weighted_f1": WARNING}
            names = {"accuracy": "Accuracy", "macro_f1": "Macro F1", "weighted_f1": "Weighted F1"}
            xs = [PRETTY[m].replace("Complement Naive Bayes", "ComplementNB")
                  .replace("Logistic Regression", "Logistic Reg.") for m in order]
            for metric in ["accuracy", "macro_f1", "weighted_f1"]:
                fig.add_bar(name=names[metric], x=xs, y=[source[m][metric] for m in order],
                            marker_color=palette[metric],
                            text=[f"{source[m][metric]:.3f}" for m in order],
                            textposition="outside", textfont=dict(size=9))
            fig.update_layout(barmode="group", yaxis_range=[0, 1])
            st.plotly_chart(chart_style(fig, height=330), use_container_width=True)
    with cR:
        with st.container(border=True):
            st.markdown("**Metrics Table**")
            rows = []
            for m in order:
                row = {"Model": PRETTY[m].replace("Logistic Regression", "Logistic Reg.")
                       .replace("Complement Naive Bayes", "ComplementNB"),
                       "Acc": source[m]["accuracy"], "Macro F1": source[m]["macro_f1"],
                       "Wt F1": source[m]["weighted_f1"]}
                rows.append(row)
            comp = pd.DataFrame(rows).set_index("Model")
            st.dataframe(comp.style.format("{:.4f}")
                         .highlight_max(axis=0, color="rgba(52,199,89,0.16)"),
                         use_container_width=True)

    # ---- winner summary ----
    st.markdown(
        f"""
        <div class="bp-card">
          <div style="font-weight:700;font-size:1.02rem;margin-bottom:.4rem;">
            Winner — {PRETTY[best]}</div>
          <div style="color:{INK};font-size:.92rem;line-height:1.55;">
            <b>{PRETTY[best]}</b> takes the highest macro-F1 ({b['macro_f1']:.3f}). The Bi-LSTM
            matches it on raw accuracy but the class-weighted linear model generalises better to
            the rare positive and neutral classes. <b>Why:</b> on a modestly sized, imbalanced
            dataset the deep model has too little data to beat well-engineered TF-IDF features.
            <b>Trade-off:</b> the classical model trains in seconds and is fully interpretable;
            the Bi-LSTM is context-aware but slower and harder to explain.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Section 5: Confusion Matrix Analysis ----
    section_heading("Confusion Matrix Analysis",
                    "Where each model's predictions land versus the truth.")
    if FM:
        m1, m2 = st.columns(2)
        with m1:
            with st.container(border=True):
                st.plotly_chart(confusion_heatmap(
                    FM["models"]["LogisticRegression"]["confusion_matrix"],
                    "Classical · Logistic Regression", ACCENT), use_container_width=True)
        with m2:
            if "BiLSTM" in FM["models"]:
                with st.container(border=True):
                    st.plotly_chart(confusion_heatmap(
                        FM["models"]["BiLSTM"]["confusion_matrix"],
                        "Deep Learning · Bi-LSTM", "#5E5CE6"), use_container_width=True)
    else:
        m1, m2 = st.columns(2)
        for col, (title, img) in zip((m1, m2),
                                     [("Classical · Logistic Regression", "cm_logreg.png"),
                                      ("Deep Learning · Bi-LSTM", "cm_lstm.png")]):
            p = os.path.join(REPORTS, img)
            if os.path.exists(p):
                with col.container(border=True):
                    st.markdown(f"**{title}**")
                    st.image(p, use_container_width=True)

    st.markdown(
        f"""
        <div class="bp-card">
          <div style="font-weight:700;font-size:1.0rem;margin-bottom:.4rem;">Interpretation</div>
          <div style="color:{INK};font-size:.9rem;line-height:1.55;">
            Both models are strongest on the <b>negative</b> class and weakest on <b>neutral</b>.
            The dominant error is <b>neutral ↔ negative</b> confusion: flat complaints such as
            “flight delayed 2 hrs” contain no overtly negative words, and short neutral questions
            can read as mild complaints. <b>Class imbalance</b> (63% negative) pulls both models
            toward the majority class, which is precisely why we report macro-F1 and apply class
            weighting during training. The hardest single case for both models is genuine
            <b>sarcasm</b> (“great, another delay”).
          </div>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 4 — ANALYTICS & INSIGHTS
# ===========================================================================
elif page == "Analytics":
    section_heading("Analytics & Insights",
                    "The data-science evidence behind the models.")

    # ---- Dataset insights / class distribution ----
    cA, cB = st.columns([1, 1])
    with cA:
        with st.container(border=True):
            st.markdown("**Class Distribution**")
            if FM:
                cd = FM["class_distribution"]
                vals = [cd[l] for l in LABELS]
            elif df is not None:
                vc = df["airline_sentiment"].value_counts()
                vals = [int(vc.get(l, 0)) for l in LABELS]
            else:
                vals = [0, 0, 0]
            fig = px.bar(x=[l.capitalize() for l in LABELS], y=vals,
                         color=[l for l in LABELS], color_discrete_map=SENT_COLORS)
            fig.update_traces(text=vals, textposition="outside")
            st.plotly_chart(chart_style(fig, height=300, legend=False), use_container_width=True)
    with cB:
        with st.container(border=True):
            st.markdown("**Dataset Insights**")
            n = FM["test_size"] if FM else (len(df) if df is not None else 0)
            st.markdown(
                f"<div style='color:{INK};font-size:.92rem;line-height:1.9;'>"
                f"Dataset&nbsp;·&nbsp;<b>Twitter US Airline Sentiment</b><br>"
                f"Labelled tweets&nbsp;·&nbsp;<b>14,640</b><br>"
                f"Held-out test set&nbsp;·&nbsp;<b>{n:,}</b><br>"
                f"Classes&nbsp;·&nbsp;<b>negative / neutral / positive</b><br>"
                f"Balance&nbsp;·&nbsp;<b>≈ 63 / 21 / 16 %</b> (imbalanced)<br>"
                f"Split&nbsp;·&nbsp;<b>80 / 20 stratified</b></div>",
                unsafe_allow_html=True)
            st.caption("People mostly tweet at airlines to complain — hence the heavy negative skew.")

    # ---- Learning curves + training diagnostics ----
    hist_p = os.path.join(REPORTS, "lstm_history.json")
    if os.path.exists(hist_p):
        h = json.load(open(hist_p))
        ep = list(range(1, len(h["accuracy"]) + 1))
        be = int(min(range(len(h["val_loss"])), key=lambda i: h["val_loss"][i]))
        d1, d2 = st.columns([1.4, 1])
        with d1:
            with st.container(border=True):
                st.markdown("**Bi-LSTM Learning Curves**")
                fig = go.Figure()
                fig.add_scatter(x=ep, y=h["accuracy"], name="Train acc",
                                line=dict(color=ACCENT, width=2.4), mode="lines+markers")
                fig.add_scatter(x=ep, y=h["val_accuracy"], name="Val acc",
                                line=dict(color=SUCCESS, width=2.4), mode="lines+markers")
                fig.add_scatter(x=ep, y=h["loss"], name="Train loss",
                                line=dict(color=INK_SOFT, width=1.6, dash="dot"), mode="lines")
                fig.add_scatter(x=ep, y=h["val_loss"], name="Val loss",
                                line=dict(color=DANGER, width=1.6, dash="dot"), mode="lines")
                fig.update_layout(xaxis_title="Epoch")
                st.plotly_chart(chart_style(fig, height=320), use_container_width=True)
        with d2:
            with st.container(border=True):
                st.markdown("**Training Diagnostics**")
                st.markdown(
                    f"<div style='color:{INK};font-size:.92rem;line-height:1.95;'>"
                    f"Epochs trained&nbsp;·&nbsp;<b>{len(h['val_loss'])}</b><br>"
                    f"Best epoch (val-loss)&nbsp;·&nbsp;<b>{be+1}</b><br>"
                    f"Best val accuracy&nbsp;·&nbsp;<b>{max(h['val_accuracy']):.1%}</b><br>"
                    f"Final train accuracy&nbsp;·&nbsp;<b>{h['accuracy'][-1]:.1%}</b></div>",
                    unsafe_allow_html=True)
                st.caption("EarlyStopping with restore_best_weights keeps the best-val-loss "
                           "epoch, so the overfit tail is never saved.")

    # ---- Word clouds ----
    wc = os.path.join(REPORTS, "wordclouds.png")
    if os.path.exists(wc):
        with st.container(border=True):
            st.markdown("**Word Clouds by Sentiment Class**")
            st.image(wc, use_container_width=True)
            st.caption("Most frequent tokens per class after cleaning — note the not_* negation "
                       "tokens surfacing in the negative cloud, proof the negation step works.")

    # ---- Key findings ----
    st.markdown(
        f"""
        <div class="bp-card">
          <div style="font-weight:700;font-size:1.0rem;margin-bottom:.5rem;">Key Findings</div>
          <ul style="margin:0;padding-left:1.1rem;color:{INK};font-size:.91rem;line-height:1.7;">
            <li>A well-tuned <b>classical model wins</b> on macro-F1 — deep learning needs more
                data to justify its complexity here.</li>
            <li><b>Macro-F1, not accuracy</b>, is the honest metric on 63%-negative data.</li>
            <li><b>Neutral</b> is the hardest class; <b>neutral ↔ negative</b> is the main error.</li>
            <li><b>Negation fusion</b> (not_good) measurably helps the bag-of-words models.</li>
            <li>One <b>shared cleaning module</b> guarantees no train/serve skew.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
