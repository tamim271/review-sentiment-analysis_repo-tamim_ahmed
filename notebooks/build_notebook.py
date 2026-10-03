"""Builds notebooks/sentiment_analysis.ipynb from the cells below.
(Kept so the notebook's structure is reviewable as plain Python in git diffs.)
Run:  python notebooks/build_notebook.py
then: jupyter nbconvert --to notebook --execute --inplace notebooks/sentiment_analysis.ipynb
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("""# Online Review Sentiment Analysis (Flipkart / Twitter)
**Tamim Ahmed · ID 905253015 · United International University, School of Business and Economics**

Three-class sentiment analysis (positive / neutral / negative) of a **synthetic** dataset of 2,000 marketplace
reviews and tweets that I generated for this project (`src/generate_dataset.py`).
The notebook follows the four deliverables of the assignment brief. All heavy lifting lives in
`src/sentiment_pipeline.py` and `src/text_utils.py`; this notebook calls those functions step by step and
shows the results."""),
    code("""import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from IPython.display import Image, display
import sentiment_pipeline as sp
from text_utils import clean_text, clean_text_demo

pd.set_option("display.max_colwidth", 120)
sp.TAB.mkdir(parents=True, exist_ok=True)
fig = lambda name: display(Image(filename=str(sp.FIG / name), width=820))"""),
    md("""## 0. The dataset
If `data/synthetic_reviews.csv` is missing, regenerate it with `python src/generate_dataset.py`
(seeded with my student ID, so it is exactly reproducible)."""),
    code("""df = sp.load_data()
print(df.shape)
df.sample(6, random_state=1)[["review_text", "product", "source", "sentiment", "rating"]]"""),
    md("## 1. Model selection and implementation\n### 1.1 Exploration: class balance, source split, review length"),
    code("""eda = sp.explore(df)
eda"""),
    code('fig("fig01_eda.png")'),
    md("""Positive reviews are the largest class (46%), neutral the smallest (24%), so the data are moderately
imbalanced. That is why I use a **stratified** split and report **macro-F1** (which weights every class equally)
alongside the weighted metrics the brief asks for. Tweets are shorter than marketplace reviews."""),
    md("""### 1.2 Preprocessing
My cleaner differs from the class demo in four deliberate ways: negations and contrast words are kept,
contractions are expanded, emojis become sentiment tokens, and elongated words are shortened."""),
    code("""examples = ["@FlipkartSupport the jar quality is NOT worth the money!! #fail 😡 https://t.co/x1",
            "Can't fault the build quality, delivery wasn't slow either 😍",
            "sooo happy with it, veryyy good"]
pd.DataFrame({"raw": examples,
              "class-demo cleaning": [clean_text_demo(t) for t in examples],
              "my cleaning": [clean_text(t) for t in examples]})"""),
    code("""df = sp.preprocess(df)
train, test = sp.split(df)
print("train:", len(train), " test:", len(test))
pd.concat([train.sentiment.value_counts(normalize=True).rename("train"),
           test.sentiment.value_counts(normalize=True).rename("test")], axis=1).round(3)"""),
    md("""### 1.3 Choosing TF-IDF settings (training data only, 5-fold CV)"""),
    code("""grid, vocab, best = sp.tune_tfidf(train)
print("vocabulary size by n-gram range:", vocab)
print("best:", best)
grid.round(3)"""),
    code('fig("fig02_tfidf_grid.png")'),
    md("""Bigrams (1,2) beat unigrams slightly because they keep phrases such as *not worth*, *very disappointed* and
*three stars*; trigrams add 2,000 mostly one-off terms and hurt. With `min_df=2` the full (1,2) vocabulary is
about 2,075 terms, so `max_features=2000` keeps essentially all of it at the same score while capping
dimensionality. **Chosen: `ngram_range=(1,2)`, `max_features=2000`, `sublinear_tf=True`, `min_df=2`.**"""),
    code("""sp.preprocessing_ablation(train).round(4)"""),
    md("Keeping negations and emojis raises cross-validated macro-F1 by about 1.4 points over the demo cleaner."),
    md("""### 1.4 Train four classifiers
* **Logistic Regression**: linear, fast, interpretable coefficients, gives probabilities.
* **Multinomial Naive Bayes**: the classic text baseline; assumes word-count-like features.
* **Linear SVM**: max-margin linear model, usually the strongest with sparse high-dimensional TF-IDF.
* **Random Forest**: a non-linear ensemble, included to test whether word interactions help."""),
    code("""results, fitted, preds = sp.train_and_evaluate(train, test)
vader = sp.vader_baseline(test)"""),
    md("## 2. Evaluation and model comparison"),
    code("""show = pd.concat([results, pd.DataFrame([{k: v for k, v in vader.items() if k != "pred"}])], ignore_index=True)
show.round(3)"""),
    code("""sp.plot_model_comparison(results, vader)
fig("fig03_model_comparison.png")"""),
    code("""best_name = results.iloc[0]["Model"]
from sklearn.metrics import classification_report
print("Best model:", best_name)
print(classification_report(test["sentiment"], preds[best_name], digits=3))
cm = sp.plot_confusion(test, preds[best_name], best_name)
fig("fig04_confusion_matrix.png")"""),
    md("""### 2.1 Error analysis
`pattern` is generator metadata (never used as a feature). It lets us see *what kind* of text the model gets
wrong, and compare with the VADER lexicon."""),
    code("""by_pattern, errors = sp.error_analysis(test, preds[best_name], vader["pred"])
by_pattern.round(3)"""),
    code("""errors[["review_text", "pattern", "sentiment", "pred"]]"""),
    code("""sp.top_terms(fitted["Logistic Regression"])"""),
    md("## 3. Sentiment trends"),
    code("""share, by_source = sp.plot_distribution_by_source(df)
fig("fig05_distribution_by_source.png")
by_source.round(1)"""),
    code("""monthly, volume = sp.plot_time_trend(df)
fig("fig06_sentiment_over_time.png")"""),
    code("""nss, aspects = sp.plot_products_and_aspects(df)
fig("fig07_products_and_aspects.png")"""),
    code("""sw = sp.plot_smartwatch(df)
fig("fig08_smartwatch_recovery.png")
sw"""),
    code("""df.groupby("sentiment")["rating"].mean().round(2)   # star rating is a sanity check, not a feature"""),
    md("""## 4. Robustness checks
**(a) Hand-written challenge set**: 36 reviews I wrote in phrasing the generator never uses.
**(b) The class sample file**: the same pipeline on `online_reviews_sentiment.csv`."""),
    code("""ch_results, ch = sp.challenge_set_eval(fitted)
ch_results.round(3)"""),
    code("""sp.class_sample_check(sp.DATA / "class_sample" / "online_reviews_sentiment.csv")"""),
    md("""The models score 96% on the held-out synthetic test set but only about 42-50% on unseen real-style
phrasing, below the untrained VADER lexicon (61%). The class sample reaches 100% because it has only 497 unique
texts in 1,000 rows, so test sentences also appear in training. Both results show that a high test score on
template-based synthetic data mostly measures how well the model learned the templates. See the report for
the discussion."""),
    md("## 5. Save the summary used in the report"),
    code("""summary = sp.main()   # re-runs everything end to end and writes outputs/tables/summary.json"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).resolve().parent / "sentiment_analysis.ipynb"
nbf.write(nb, out)
print("wrote", out)
