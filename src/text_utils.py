"""
text_utils.py
-------------
Text preprocessing for review / tweet sentiment analysis.

Design choices (and why they differ from the class demo):
  1. Negations and contrast words are KEPT. NLTK's English stopword list
     contains "not", "no", "nor", "never", "but", "don't", "isn't" ... Removing
     them turns "not good at all" into "good", which flips the meaning.
  2. Contractions are expanded first ("doesn't" -> "does not") so that the
     negation survives as a separate token that TF-IDF bigrams can pair with
     the next word ("not good", "not worth").
  3. Emojis are converted to sentiment tokens (emo_pos / emo_neg / emo_neu)
     instead of being deleted with the punctuation, because on Twitter they
     often carry the clearest sentiment signal.
  4. Hashtags keep their word ("#fail" -> "fail"); only the "#" is removed.
     URLs and @mentions are removed because they carry no sentiment.
  5. Elongated words are shortened ("sooo" -> "soo", "veryyy" -> "veryy") so
     spelling variants collapse to fewer vocabulary entries.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download the NLTK resources on first run if they are missing
for _res, _path in [("stopwords", "corpora/stopwords"), ("punkt", "tokenizers/punkt"),
                    ("punkt_tab", "tokenizers/punkt_tab"), ("vader_lexicon", "sentiment/vader_lexicon.zip")]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_res, quiet=True)

# Words that change or carry sentiment and must survive stopword removal
KEEP_WORDS = {
    "not", "no", "nor", "never", "but", "against", "only", "very", "too",
    "don", "doesn", "didn", "isn", "wasn", "aren", "weren", "won", "wouldn",
    "couldn", "shouldn", "hasn", "haven", "hadn", "mightn", "mustn", "needn", "ain",
}
STOP_WORDS = set(stopwords.words("english")) - KEEP_WORDS
DEFAULT_STOP_WORDS = set(stopwords.words("english"))   # used for the ablation test

EMOJI_MAP = {
    "😍": " emo_pos ", "🔥": " emo_pos ", "👍": " emo_pos ", "❤️": " emo_pos ", "❤": " emo_pos ", "😊": " emo_pos ",
    "😡": " emo_neg ", "👎": " emo_neg ", "😤": " emo_neg ", "🙄": " emo_neg ", "💔": " emo_neg ",
    "🤔": " emo_neu ", "📦": " emo_neu ", "🙂": " emo_neu ",
}

CONTRACTIONS = [
    (r"\bcan't\b", "can not"), (r"\bwon't\b", "will not"), (r"\bain't\b", "is not"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'s\b", ""), (r"'ve\b", " have"),
    (r"'ll\b", " will"), (r"'d\b", " would"), (r"'m\b", " am"),
]

URL_RE = re.compile(r"http\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
ELONGATION_RE = re.compile(r"(\w)\1{2,}")
PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation + "’‘“”"})


def clean_text(text: str, keep_negations: bool = True) -> str:
    """Clean one review and return a space-separated string of tokens."""
    text = str(text)
    for emo, token in EMOJI_MAP.items():                 # 1. emojis -> tokens
        text = text.replace(emo, token)
    text = text.lower()                                  # 2. lowercase
    text = URL_RE.sub(" ", text)                         # 3. URLs
    text = MENTION_RE.sub(" ", text)                     # 4. @mentions
    text = text.replace("#", " ")                        # 5. keep hashtag word
    for pattern, repl in CONTRACTIONS:                   # 6. expand contractions
        text = re.sub(pattern, repl, text)
    text = ELONGATION_RE.sub(r"\1\1", text)              # 7. sooo -> soo
    text = text.translate(PUNCT_TABLE)                   # 8. punctuation (keeps emo_pos underscores out of the way)
    text = text.replace(" emo pos ", " emo_pos ").replace(" emo neg ", " emo_neg ").replace(" emo neu ", " emo_neu ")
    tokens = word_tokenize(text)                         # 9. tokenize
    stops = STOP_WORDS if keep_negations else DEFAULT_STOP_WORDS
    tokens = [t for t in tokens
              if (t.isalpha() or t.startswith("emo_")) and t not in stops and len(t) > 1]
    return " ".join(tokens)


def clean_text_demo(text: str) -> str:
    """The class-demo cleaner (standard stopwords, emojis dropped). Kept only
    so the effect of my preprocessing choices can be measured fairly."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\w+|#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    return " ".join(t for t in tokens if t.isalpha() and t not in DEFAULT_STOP_WORDS)
