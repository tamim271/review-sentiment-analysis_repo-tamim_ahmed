# Online Review Sentiment Analysis (Flipkart / Twitter)

**Author:** Tamim Ahmed · **Student ID:** 905253015
**Course project:** Real-Life Project: NLP & Sentiment Analysis, School of Business and Economics, United International University

This project builds a complete three-class sentiment analysis pipeline (positive / neutral / negative) for online
product reviews and social-media posts. Because the class sample file is built from about 45 repeated templates,
I wrote my own generator that produces a more realistic **synthetic dataset of 2,000 Flipkart-style reviews and
tweets**, including negation, mixed opinions, sarcasm, "wait-and-see" posts and a little label noise. The
pipeline cleans the text, converts it to TF-IDF features (settings chosen by cross-validation), trains four
classifiers, compares them against a VADER lexicon baseline, and analyses sentiment trends over time, by source,
by product and by complaint topic. A Linear SVM performs best (96.0% accuracy, 0.954 macro-F1 on a held-out test
set). A hand-written challenge set shows that the models do not carry over well to phrasing they have never seen.

## Repository structure

```
├── data/
│   ├── synthetic_reviews.csv / .xlsx   # my generated dataset (2,000 rows)
│   ├── challenge_set.csv               # 36 hand-written reviews for a robustness check
│   └── class_sample/                   # the instructor's sample file (used for comparison only)
├── src/
│   ├── generate_dataset.py             # synthetic data generator (seeded, reproducible)
│   ├── text_utils.py                   # preprocessing (negation-aware cleaning)
│   └── sentiment_pipeline.py           # TF-IDF tuning, models, evaluation, trend charts
├── notebooks/
│   ├── sentiment_analysis.ipynb        # step-by-step analysis with all outputs (executed)
│   └── build_notebook.py               # builds the notebook from plain Python
├── outputs/
│   ├── figures/                        # fig01 … fig08 (PNG)
│   └── tables/                         # metrics, confusion matrix, errors, summary.json
├── report/                             # written report (Word + PDF)
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. (optional) regenerate the dataset; the seed makes it identical every time
python src/generate_dataset.py

# 3. run the full pipeline: writes every figure and table to outputs/
python src/sentiment_pipeline.py

# 4. or work through it step by step
jupyter notebook notebooks/sentiment_analysis.ipynb
```

NLTK resources (stopwords, punkt, VADER lexicon) are downloaded automatically on first run.

## Dataset

| Column | Description |
|---|---|
| `review_id` | 1–2,000, ordered by date |
| `review_text` | the review or tweet |
| `product` | one of 12 products (electronics, kitchen, home & furniture, fashion) |
| `category` | product category |
| `aspect` | main topic, e.g. battery, delivery, customer_service (used for trend analysis only) |
| `sentiment` | label: positive / neutral / negative |
| `source` | Flipkart (marketplace review) or Twitter (social post) |
| `date` | Oct 2025 – Sep 2026 |
| `rating` | 1–5 stars, correlated with sentiment (sanity check only, never a model feature) |
| `pattern` | generator metadata: plain / mixed / negation / sarcasm / wait_and_see / noisy_label (error analysis only) |

Class balance: 46% positive, 30% negative, 24% neutral. 60% Flipkart, 40% Twitter. All 2,000 texts are unique.
The generator embeds three "business stories" for the trend analysis to find: a festive-sale delivery crunch
(Oct–Nov 2025), a buggy smartwatch launch fixed by a firmware update (Feb–May 2026), and monsoon delivery
delays (Jul–Aug 2026). **All data are synthetic; no real customers or posts are included.**

## Key results

Test set: 400 reviews (stratified 20%). TF-IDF: unigrams + bigrams, `max_features=2000`, `sublinear_tf`, `min_df=2`.

| Model | Accuracy | Precision (w) | Recall (w) | F1 (weighted) | F1 (macro) | 5-fold CV macro-F1 |
|---|---|---|---|---|---|---|
| **Linear SVM** | **0.960** | **0.960** | **0.960** | **0.960** | **0.954** | **0.961 ± 0.019** |
| Random Forest | 0.945 | 0.945 | 0.945 | 0.945 | 0.939 | 0.933 ± 0.020 |
| Logistic Regression | 0.935 | 0.935 | 0.935 | 0.934 | 0.928 | 0.947 ± 0.019 |
| Multinomial NB | 0.902 | 0.902 | 0.902 | 0.902 | 0.890 | 0.899 ± 0.026 |
| VADER lexicon (no training) | 0.658 | 0.649 | 0.658 | 0.644 | 0.610 | n/a |

* **Neutral is the hardest class** (F1 0.915); the most common error is neutral predicted as positive (6 of 95).
* 11 of the SVM's 16 test errors are reviews whose label was deliberately flipped (label noise); on those, the
  model's prediction matches the text, not the label.
* Keeping negations and emojis in preprocessing raises CV macro-F1 from 0.933 to 0.947 compared with the class-demo cleaner.
* **Generalisation warning:** on 36 hand-written reviews in new phrasing, accuracy drops to 42–50%, below VADER (61%).
  On the class sample file the same pipeline scores 100% because half its rows are duplicates.
* **Trends:** Twitter is far more negative than Flipkart (43% vs 21% negative); delivery is the top complaint on both
  platforms, and customer service makes up 28% of Twitter complaints. Net sentiment ranges from +50 (air fryer) to −6
  (office chair). The smartwatch went from 76% negative before the firmware fix to 14% after.

See `report/` for the full interpretation and business recommendations.
