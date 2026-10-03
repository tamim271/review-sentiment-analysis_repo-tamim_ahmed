"""
sentiment_pipeline.py
---------------------
End-to-end sentiment analysis of synthetic Flipkart reviews and tweets.

Steps (each is a function, so the notebook can call them one at a time):
  1. load + explore the data
  2. preprocess text (see text_utils.py for the design choices)
  3. choose TF-IDF settings with cross-validation on the TRAINING set only
  4. train four classifiers (+ a lexicon baseline) on a stratified split
  5. evaluate: accuracy / precision / recall / F1, confusion matrix, errors
  6. sentiment trends: over time, by source, by product, by aspect
  7. robustness: hand-written challenge set + the class sample dataset

Run everything from the repository root:
    python src/sentiment_pipeline.py
Figures go to outputs/figures/, tables to outputs/tables/.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from text_utils import clean_text, clean_text_demo  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
RANDOM_STATE = 42
LABELS = ["negative", "neutral", "positive"]

# --- chart styling ---------------------------------------------------------
# Sentiment is a polarity, so it gets a diverging scheme: red <-> grey <-> blue.
SENT_COLORS = {"negative": "#e34948", "neutral": "#a3a19b", "positive": "#2a78d6"}
# Categorical slots (validated colour-blind-safe order) for model identity.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e2dd"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _hgrid(ax):
    """Horizontal bar charts read against vertical gridlines."""
    ax.grid(False, axis="y")
    ax.grid(True, axis="x")


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, facecolor="white")
    plt.close(fig)
    return FIG / name


# ===========================================================================
# 1. Load and explore
# ===========================================================================
def load_data(path=DATA / "synthetic_reviews.csv") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df["n_words"] = df["review_text"].str.split().str.len()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    return df


def explore(df: pd.DataFrame) -> dict:
    """Return the headline EDA numbers and save the EDA figure."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    counts = df["sentiment"].value_counts().reindex(LABELS)
    axes[0].bar(counts.index, counts.values, color=[SENT_COLORS[l] for l in LABELS], width=0.6)
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 15, f"{v}\n({v / len(df):.0%})", ha="center", va="bottom", fontsize=8, color=INK)
    axes[0].set_ylim(0, counts.max() * 1.3)
    axes[0].set_title("Class balance")
    axes[0].set_ylabel("Reviews")

    src = df["source"].value_counts()
    axes[1].bar(src.index, src.values, color=CAT[0], width=0.5)
    for i, v in enumerate(src.values):
        axes[1].text(i, v + 15, f"{v}", ha="center", fontsize=8, color=INK)
    axes[1].set_ylim(0, src.max() * 1.2)
    axes[1].set_title("Reviews by source")

    for i, s in enumerate(["Flipkart", "Twitter"]):
        axes[2].hist(df.loc[df.source == s, "n_words"], bins=range(0, 46, 2), alpha=0.75,
                     color=CAT[i], label=s, edgecolor="white", linewidth=0.8)
    axes[2].set_title("Review length (words)")
    axes[2].legend()
    fig.tight_layout()
    _save(fig, "fig01_eda.png")

    return {
        "n_reviews": len(df),
        "class_share": (counts / len(df)).round(3).to_dict(),
        "source_counts": src.to_dict(),
        "median_words": df.groupby("source")["n_words"].median().to_dict(),
        "date_range": [str(df.date.min().date()), str(df.date.max().date())],
        "n_products": df["product"].nunique(),
        "unique_texts": int(df["review_text"].nunique()),
    }


# ===========================================================================
# 2. Preprocess and split
# ===========================================================================
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_text"] = df["review_text"].apply(clean_text)
    df["clean_text_demo"] = df["review_text"].apply(clean_text_demo)
    return df


def split(df: pd.DataFrame):
    """80/20 split, stratified so each class keeps its share in both parts."""
    return train_test_split(df, test_size=0.2, stratify=df["sentiment"], random_state=RANDOM_STATE)


def cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def tfidf(**kw):
    # sublinear_tf dampens repeated words; min_df=2 drops one-off tokens (typos, URL fragments)
    params = dict(sublinear_tf=True, min_df=2)
    params.update(kw)
    return TfidfVectorizer(**params)


# ===========================================================================
# 3. Choose TF-IDF settings by cross-validation (training data only)
# ===========================================================================
def tune_tfidf(train: pd.DataFrame):
    """Grid over max_features x ngram_range, scored by 5-fold macro-F1 with
    Logistic Regression. The vectoriser sits inside the Pipeline, so each fold
    learns its vocabulary from its own training part (no leakage)."""
    pipe = Pipeline([("tfidf", tfidf()), ("clf", LogisticRegression(max_iter=2000))])
    grid = {"tfidf__max_features": [500, 1000, 2000, 5000, None],
            "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)]}
    gs = GridSearchCV(pipe, grid, cv=cv(), scoring="f1_macro", n_jobs=-1)
    gs.fit(train["clean_text"], train["sentiment"])
    res = pd.DataFrame(gs.cv_results_)
    res["max_features"] = res["param_tfidf__max_features"].apply(lambda v: "All" if v is None else str(int(v)))
    res["ngram_range"] = res["param_tfidf__ngram_range"].astype(str)
    table = res.pivot(index="ngram_range", columns="max_features", values="mean_test_score")
    table = table[["500", "1000", "2000", "5000", "All"]]

    # vocabulary sizes help justify the choice
    vocab = {str(ng): len(tfidf(ngram_range=ng).fit(train["clean_text"]).vocabulary_)
             for ng in [(1, 1), (1, 2), (1, 3)]}

    fig, ax = plt.subplots(figsize=(5.2, 2.3))
    im = ax.imshow(table.values, cmap="Blues", vmin=table.values.min() - 0.01, vmax=table.values.max())
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), [f"{r}  ({vocab[r]:,} terms)" for r in table.index])
    ax.grid(False)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if im.norm(v) > 0.55 else INK)
    ax.set_xlabel("max_features")
    ax.set_ylabel("n-gram range (vocabulary)")
    ax.set_title("TF-IDF tuning: 5-fold CV macro-F1 (Logistic Regression)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    _save(fig, "fig02_tfidf_grid.png")
    table.to_csv(TAB / "tfidf_grid_cv_macro_f1.csv")
    return table, vocab, gs.best_params_


def preprocessing_ablation(train: pd.DataFrame, ngram=(1, 2), max_features=2000) -> pd.DataFrame:
    """Does keeping negations/emojis actually help? Compare with the demo cleaner."""
    rows = []
    for name, col in [("Class-demo cleaning (standard stopwords)", "clean_text_demo"),
                      ("My cleaning (negations + emojis kept)", "clean_text")]:
        pipe = Pipeline([("tfidf", tfidf(ngram_range=ngram, max_features=max_features)),
                         ("clf", LogisticRegression(max_iter=2000))])
        s = cross_val_score(pipe, train[col], train["sentiment"], cv=cv(), scoring="f1_macro")
        rows.append({"Preprocessing": name, "CV macro-F1": s.mean(), "SD": s.std()})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "preprocessing_ablation.csv", index=False)
    return out


# ===========================================================================
# 4. Models
# ===========================================================================
def build_models():
    """Four classic text classifiers that suit sparse, high-dimensional TF-IDF input."""
    return {
        # linear, well calibrated probabilities, coefficients are interpretable
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        # generative baseline built for word-count style features; very fast
        "Multinomial NB": MultinomialNB(alpha=0.5),
        # max-margin linear model; usually the strongest on sparse text features
        "Linear SVM": LinearSVC(C=1.0),
        # non-linear ensemble; can capture word interactions but sparse data is hard for trees
        "Random Forest": RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
    }


def train_and_evaluate(train, test, ngram=(1, 2), max_features=2000):
    """Fit each model in a TF-IDF pipeline, evaluate on the SAME held-out test set."""
    fitted, preds, rows = {}, {}, []
    for name, clf in build_models().items():
        pipe = Pipeline([("tfidf", tfidf(ngram_range=ngram, max_features=max_features)), ("clf", clf)])
        cv_f1 = cross_val_score(pipe, train["clean_text"], train["sentiment"], cv=cv(), scoring="f1_macro")
        pipe.fit(train["clean_text"], train["sentiment"])
        p = pipe.predict(test["clean_text"])
        prec, rec, f1, _ = precision_recall_fscore_support(test["sentiment"], p, average="weighted", zero_division=0)
        rows.append({"Model": name, "Accuracy": accuracy_score(test["sentiment"], p), "Precision": prec,
                     "Recall": rec, "F1 (weighted)": f1,
                     "F1 (macro)": f1_score(test["sentiment"], p, average="macro"),
                     "CV macro-F1 (mean)": cv_f1.mean(), "CV macro-F1 (SD)": cv_f1.std()})
        fitted[name], preds[name] = pipe, p
    results = pd.DataFrame(rows).sort_values(["F1 (macro)", "F1 (weighted)"], ascending=False).reset_index(drop=True)
    results.to_csv(TAB / "model_comparison.csv", index=False)
    return results, fitted, preds


def vader_label(texts: pd.Series) -> pd.Series:
    """VADER compound score -> label, using the standard +/-0.05 thresholds."""
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    def label(t):
        c = sia.polarity_scores(t)["compound"]
        return "positive" if c >= 0.05 else "negative" if c <= -0.05 else "neutral"

    return texts.apply(label)


def vader_baseline(test) -> dict:
    """Rule-based lexicon baseline (no training). Shows what a supervised model adds."""
    p = vader_label(test["review_text"])
    prec, rec, f1, _ = precision_recall_fscore_support(test["sentiment"], p, average="weighted", zero_division=0)
    return {"Model": "VADER lexicon (baseline)", "Accuracy": accuracy_score(test["sentiment"], p),
            "Precision": prec, "Recall": rec, "F1 (weighted)": f1,
            "F1 (macro)": f1_score(test["sentiment"], p, average="macro"), "pred": p}


def plot_model_comparison(results: pd.DataFrame, vader: dict):
    metrics = ["Accuracy", "Precision", "Recall", "F1 (weighted)", "F1 (macro)"]
    allm = pd.concat([results, pd.DataFrame([{k: v for k, v in vader.items() if k != "pred"}])], ignore_index=True)
    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    n = len(allm)
    w = 0.8 / n
    x = np.arange(len(metrics))
    for i, row in allm.iterrows():
        vals = row[metrics].astype(float).values
        ax.bar(x + (i - (n - 1) / 2) * w, vals, width=w * 0.88, color=CAT[i], label=row["Model"])
    best = allm.iloc[0]
    for j, m in enumerate(metrics):
        ax.text(x[j] - (n - 1) / 2 * w, best[m] + 0.01, f"{best[m]:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score on held-out test set (n = 400)")
    ax.set_title("Model comparison (y-axis starts at 0.5)")
    ax.legend(ncol=3, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    _save(fig, "fig03_model_comparison.png")


def plot_confusion(test, pred, name):
    cm = confusion_matrix(test["sentiment"], pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.imshow(cm, cmap="Blues")
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > cm.max() / 2 else INK)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {name}")
    _save(fig, "fig04_confusion_matrix.png")
    pd.DataFrame(cm, index=[f"actual_{l}" for l in LABELS], columns=[f"pred_{l}" for l in LABELS]) \
        .to_csv(TAB / "confusion_matrix_best.csv")
    return cm


def error_analysis(test, pred, vader_pred=None):
    """Accuracy by text pattern + a table of every misclassified review."""
    t = test.assign(pred=pred)
    t["correct"] = t["pred"] == t["sentiment"]
    by = t.groupby("pattern").agg(n=("correct", "size"), accuracy=("correct", "mean"))
    if vader_pred is not None:
        t["vader_correct"] = vader_pred.values == t["sentiment"].values
        by["vader_accuracy"] = t.groupby("pattern")["vader_correct"].mean()
    by = by.sort_values("accuracy")
    errs = t.loc[~t["correct"], ["review_id", "review_text", "source", "pattern", "sentiment", "pred"]]
    by.to_csv(TAB / "accuracy_by_pattern.csv")
    errs.to_csv(TAB / "misclassified_reviews.csv", index=False)
    return by, errs


def top_terms(pipe, k=8) -> pd.DataFrame:
    """Most influential n-grams per class for a linear model."""
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    names = np.array(vec.get_feature_names_out())
    out = {}
    for i, c in enumerate(clf.classes_):
        out[c] = names[np.argsort(clf.coef_[i])[::-1][:k]]
    df = pd.DataFrame(out)
    df.to_csv(TAB / "top_terms_per_class.csv", index=False)
    return df


# ===========================================================================
# 5. Sentiment trends
# ===========================================================================
def plot_distribution_by_source(df):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.6), gridspec_kw={"width_ratios": [1, 1.3]})
    share = df["sentiment"].value_counts(normalize=True).reindex(LABELS) * 100
    axes[0].barh(LABELS, share.values, color=[SENT_COLORS[l] for l in LABELS], height=0.55)
    for i, v in enumerate(share.values):
        axes[0].text(v + 1, i, f"{v:.0f}%", va="center", fontsize=8, color=INK)
    axes[0].set_xlim(0, 60)
    axes[0].set_title("Overall sentiment (all 2,000 reviews)")
    _hgrid(axes[0])
    axes[0].set_xlabel("% of reviews")

    ct = pd.crosstab(df["source"], df["sentiment"], normalize="index").reindex(columns=LABELS) * 100
    left = np.zeros(len(ct))
    for l in LABELS:
        axes[1].barh(ct.index, ct[l], left=left, color=SENT_COLORS[l], height=0.55, label=l,
                     edgecolor="white", linewidth=1.5)
        for i, (v, lft) in enumerate(zip(ct[l], left)):
            if v > 8:
                axes[1].text(lft + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8, color="white")
        left += ct[l].values
    axes[1].set_xlim(0, 100)
    axes[1].set_title("Sentiment share by source")
    axes[1].set_xlabel("% of reviews from that source")
    axes[1].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=8)
    axes[1].grid(False)
    fig.tight_layout()
    _save(fig, "fig05_distribution_by_source.png")
    return share, ct


def plot_time_trend(df):
    m = pd.crosstab(df["month"], df["sentiment"]).reindex(columns=LABELS)
    share = m.div(m.sum(axis=1), axis=0) * 100
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 4.4), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    for l in LABELS:
        axes[0].plot(share.index, share[l], marker="o", markersize=4, linewidth=2, color=SENT_COLORS[l], label=l)
        axes[0].text(share.index[-1] + pd.Timedelta(days=8), share[l].iloc[-1], l, va="center",
                     fontsize=8, color=INK2)
    for d, lab in [("2025-11-01", "Festive sale\n(delivery strain)"), ("2026-02-01", "Smartwatch 2\nlaunch"),
                   ("2026-05-01", "Firmware\nfix"), ("2026-07-15", "Monsoon\ndelays")]:
        axes[0].axvline(pd.Timestamp(d), color=GRID, linewidth=1.2, zorder=0)
        axes[0].text(pd.Timestamp(d), 63, lab, ha="center", va="top", fontsize=7, color=INK2)
    axes[0].set_ylim(0, 65)
    axes[0].set_ylabel("% of month's reviews")
    axes[0].set_title("Sentiment over time (monthly share)")
    axes[0].legend(ncol=3, loc="lower left", fontsize=8)
    axes[1].bar(m.index, m.sum(axis=1), width=20, color=CAT[0])
    axes[1].set_ylabel("Reviews")
    axes[1].set_title("Monthly review volume", fontsize=9)
    axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
    fig.tight_layout()
    _save(fig, "fig06_sentiment_over_time.png")
    share.to_csv(TAB / "monthly_sentiment_share.csv")
    return share, m.sum(axis=1)


def net_sentiment(frame):
    """Net sentiment score = % positive - % negative (range -100 .. +100)."""
    s = frame["sentiment"].value_counts(normalize=True)
    return 100 * (s.get("positive", 0) - s.get("negative", 0))


def plot_products_and_aspects(df):
    nss = df.groupby("product").apply(net_sentiment).sort_values()
    n = df["product"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
    colors = [SENT_COLORS["negative"] if v < 0 else SENT_COLORS["positive"] for v in nss.values]
    axes[0].barh(nss.index, nss.values, color=colors, height=0.6)
    for i, (p, v) in enumerate(nss.items()):
        axes[0].text(v + (1 if v >= 0 else -1), i, f"{v:+.0f}", va="center", ha="left" if v >= 0 else "right",
                     fontsize=7.5, color=INK)
    axes[0].axvline(0, color=INK2, linewidth=0.8)
    axes[0].set_title("Net sentiment by product\n(% positive − % negative)")
    axes[0].set_xlim(min(nss.min() - 8, -10), nss.max() + 10)
    axes[0].tick_params(axis="y", labelsize=7.5)
    _hgrid(axes[0])

    neg = df[df["sentiment"] == "negative"]
    asp = pd.crosstab(neg["aspect"], neg["source"], normalize="columns") * 100
    pretty = {"customer_service": "customer service", "value": "price / value"}
    asp = asp.rename(index=lambda a: pretty.get(a, a))
    asp = asp.loc[asp.sum(axis=1).sort_values().index].tail(8)
    y = np.arange(len(asp))
    for i, s in enumerate(["Flipkart", "Twitter"]):
        axes[1].barh(y + (0.5 - i) * 0.38, asp[s], height=0.34, color=CAT[i], label=s)
    axes[1].set_yticks(y, asp.index)
    axes[1].set_title("What negative reviews are about\n(top 8 aspects, % of each source's negatives)")
    axes[1].set_xlabel("% of negative reviews")
    axes[1].legend(loc="lower right", fontsize=8)
    _hgrid(axes[1])
    fig.tight_layout()
    _save(fig, "fig07_products_and_aspects.png")
    nss.to_frame("net_sentiment").assign(n_reviews=n).to_csv(TAB / "net_sentiment_by_product.csv")
    asp.to_csv(TAB / "negative_aspects_by_source.csv")
    return nss, asp


def plot_smartwatch(df, product="PulseFit Smartwatch 2"):
    d = df[df["product"] == product]
    m = pd.crosstab(d["month"], d["sentiment"]).reindex(columns=LABELS, fill_value=0)
    share = m.div(m.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for l in ["negative", "positive"]:
        ax.plot(share.index, share[l], marker="o", markersize=4, linewidth=2, color=SENT_COLORS[l], label=l)
    ax.axvspan(pd.Timestamp("2026-04-20"), pd.Timestamp("2026-05-10"), color=GRID, alpha=0.7, lw=0)
    ax.text(pd.Timestamp("2026-05-03"), 95, "firmware fix", ha="center", va="top", fontsize=7.5, color=INK2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of month's reviews")
    ax.set_title(f"{product}: launch bug and recovery")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    _save(fig, "fig08_smartwatch_recovery.png")
    before = d[d["date"] < "2026-05-01"]
    after = d[d["date"] >= "2026-05-01"]
    return {"n": len(d), "neg_before": (before.sentiment == "negative").mean(),
            "neg_after": (after.sentiment == "negative").mean(),
            "nss_before": net_sentiment(before), "nss_after": net_sentiment(after)}


# ===========================================================================
# 6. Robustness checks
# ===========================================================================
def challenge_set_eval(fitted: dict) -> pd.DataFrame:
    """Hand-written reviews in phrasing the generator never used."""
    ch = pd.read_csv(DATA / "challenge_set.csv")
    ch["clean_text"] = ch["review_text"].apply(clean_text)
    rows = []
    for name, pipe in fitted.items():
        p = pipe.predict(ch["clean_text"])
        rows.append({"Model": name, "Accuracy": accuracy_score(ch["sentiment"], p),
                     "F1 (macro)": f1_score(ch["sentiment"], p, average="macro")})
        ch[f"pred_{name}"] = p
    v = vader_label(ch["review_text"])
    rows.append({"Model": "VADER lexicon (baseline)", "Accuracy": accuracy_score(ch["sentiment"], v),
                 "F1 (macro)": f1_score(ch["sentiment"], v, average="macro")})
    ch["pred_VADER"] = v
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "challenge_set_results.csv", index=False)
    ch.to_csv(TAB / "challenge_set_predictions.csv", index=False)
    return out, ch


def class_sample_check(sample_path) -> float | None:
    """Same pipeline on the instructor's sample file: shows how template-based
    data inflates scores."""
    if not Path(sample_path).exists():
        return None
    s = pd.read_csv(sample_path)
    s["clean_text"] = s["review_text"].apply(clean_text)
    a, b = train_test_split(s, test_size=0.2, stratify=s["sentiment"], random_state=RANDOM_STATE)
    pipe = Pipeline([("tfidf", tfidf(ngram_range=(1, 2))), ("clf", LinearSVC())]).fit(a["clean_text"], a["sentiment"])
    return {"accuracy": accuracy_score(b["sentiment"], pipe.predict(b["clean_text"])),
            "unique_texts": int(s["review_text"].nunique()), "n": len(s)}


# ===========================================================================
# main
# ===========================================================================
def main():
    TAB.mkdir(parents=True, exist_ok=True)
    df = preprocess(load_data())
    summary = {"eda": explore(df)}
    train, test = split(df)
    summary["split"] = {"train": len(train), "test": len(test),
                        "test_class_counts": test["sentiment"].value_counts().to_dict()}

    grid, vocab, best = tune_tfidf(train)
    summary["tfidf"] = {"grid": grid.round(4).to_dict(), "vocab": vocab, "grid_best": str(best)}
    abl = preprocessing_ablation(train)
    summary["ablation"] = abl.round(4).to_dict("records")

    results, fitted, preds = train_and_evaluate(train, test)
    vader = vader_baseline(test)
    plot_model_comparison(results, vader)
    best_name = results.iloc[0]["Model"]
    cm = plot_confusion(test, preds[best_name], best_name)
    by_pattern, errs = error_analysis(test, preds[best_name], vader["pred"])
    report = classification_report(test["sentiment"], preds[best_name], output_dict=True)
    summary["models"] = results.round(4).to_dict("records")
    summary["vader"] = {k: round(v, 4) for k, v in vader.items() if k not in ("pred", "Model")}
    summary["best_model"] = best_name
    summary["best_report"] = report
    summary["confusion_matrix"] = cm.tolist()
    summary["accuracy_by_pattern"] = by_pattern.round(3).reset_index().to_dict("records")
    summary["top_terms"] = top_terms(fitted["Logistic Regression"]).to_dict("list")

    share, ct = plot_distribution_by_source(df)
    summary["overall_share"] = share.round(1).to_dict()
    summary["source_share"] = ct.round(1).to_dict("index")
    monthly, volume = plot_time_trend(df)
    summary["monthly_negative"] = {str(k.date()): round(v, 1) for k, v in monthly["negative"].items()}
    summary["monthly_positive"] = {str(k.date()): round(v, 1) for k, v in monthly["positive"].items()}
    summary["monthly_volume"] = {str(k.date()): int(v) for k, v in volume.items()}
    nss, asp = plot_products_and_aspects(df)
    summary["net_sentiment_by_product"] = nss.round(1).to_dict()
    summary["negative_aspects_by_source"] = asp.round(1).to_dict("index")
    summary["smartwatch"] = plot_smartwatch(df)
    summary["rating_by_sentiment"] = df.groupby("sentiment")["rating"].mean().round(2).to_dict()

    ch, _ = challenge_set_eval(fitted)
    summary["challenge_set"] = ch.round(3).to_dict("records")
    summary["class_sample"] = class_sample_check(DATA / "class_sample" / "online_reviews_sentiment.csv")

    with open(TAB / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(results.round(3).to_string(index=False))
    print("Best model:", best_name)
    return summary


if __name__ == "__main__":
    main()
