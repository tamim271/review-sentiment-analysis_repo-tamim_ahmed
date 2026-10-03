"""
generate_dataset.py
-------------------
Generates a synthetic dataset of e-commerce product reviews (Flipkart-style)
and social-media posts (Twitter-style) for three-class sentiment analysis.

Why build a new generator instead of reusing the class sample?
The class sample (online_reviews_sentiment.csv) is built from ~45 fixed
sentence templates, so the same sentence appears many times and every model
scores ~100%. That is fine for a demo but tells us nothing about how a model
copes with real review language. This generator is compositional: it combines
product-specific features, service aspects (delivery, packaging, price,
customer care, returns), openers, closers and platform-specific noise, and it
deliberately injects the things that make real sentiment analysis hard:

  * negation          ("not bad at all", "not worth the money")
  * mixed opinions    ("camera is great but battery is terrible")
  * sarcasm           ("Wow, battery lasts a whole two hours. Brilliant.")
  * wait-and-see text (neutral posts with no opinion yet)
  * label noise       (~2% of labels flipped to an adjacent class, mimicking
                       annotator disagreement)

It also embeds three "business stories" for the trend analysis to discover:
  1. A festive sale in Oct-Nov 2025 -> volume spike and delivery complaints.
  2. PulseFit Smartwatch 2 launched Feb 2026 with a battery/sync bug ->
     negative spike Feb-Apr, firmware fix in May -> recovery.
  3. Monsoon season (Jul-Aug 2026) -> another rise in delivery complaints.

Columns
  review_id, review_text, product, category, aspect, sentiment, source,
  date, rating, pattern
`pattern` is generator metadata (plain / mixed / negation / sarcasm /
wait_and_see / noisy_label). It is NEVER used as a model feature; it is only
used afterwards to diagnose which kinds of text the models get wrong.

Usage:  python src/generate_dataset.py            (writes to data/)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 905253015          # student ID, so the dataset is exactly reproducible
N_REVIEWS = 2000
TWITTER_SHARE = 0.40
LABEL_NOISE = 0.02

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# 1. Products and the features people talk about for each one
# ---------------------------------------------------------------------------
PRODUCTS = {
    # name: (category, {aspect_key: feature noun}, base sentiment tilt)
    "Nova 5G Smartphone":   ("Electronics", {"battery": "battery life", "performance": "performance",
                                             "camera": "camera", "build": "build quality"}, 0.00),
    "PulseFit Smartwatch 2": ("Electronics", {"battery": "battery backup", "sync": "app sync",
                                              "tracking": "step and heart-rate tracking",
                                              "display": "display"}, 0.00),
    "AirBeat Earbuds":      ("Electronics", {"sound": "sound quality", "battery": "battery life",
                                             "connectivity": "Bluetooth connection",
                                             "comfort": "fit in the ear"}, 0.05),
    "BoomBox Speaker":      ("Electronics", {"sound": "bass", "battery": "battery backup",
                                             "build": "build quality", "connectivity": "pairing"}, 0.08),
    "PowerCore Power Bank": ("Electronics", {"battery": "capacity", "charging": "charging speed",
                                             "build": "build quality", "heat": "heat management"}, 0.00),
    "ProGlide Gaming Mouse": ("Electronics", {"performance": "sensor accuracy", "build": "click feel",
                                              "comfort": "grip", "software": "RGB software"}, 0.05),
    "CrispAir Air Fryer":   ("Kitchen", {"performance": "cooking performance", "cleaning": "non-stick basket",
                                         "noise": "noise level", "build": "build quality"}, 0.15),
    "BlendPro Blender":     ("Kitchen", {"performance": "blending power", "noise": "motor noise",
                                         "build": "jar quality", "cleaning": "cleaning"}, -0.03),
    "ErgoMax Office Chair": ("Home & Furniture", {"comfort": "back support", "assembly": "assembly",
                                                  "build": "build quality", "adjust": "height adjustment"}, -0.12),
    "Lumi LED Desk Lamp":   ("Home & Furniture", {"brightness": "brightness", "build": "build quality",
                                                  "adjust": "dimmer control", "design": "design"}, 0.06),
    "StrideRun Running Shoes": ("Fashion", {"comfort": "cushioning", "fit": "size and fit",
                                            "durability": "sole durability", "design": "look"}, 0.04),
    "UrbanPack Laptop Backpack": ("Fashion", {"durability": "stitching", "space": "storage space",
                                              "comfort": "shoulder padding", "design": "design"}, 0.06),
}
PULSEFIT = "PulseFit Smartwatch 2"
PULSEFIT_LAUNCH = pd.Timestamp("2026-02-01")
FIRMWARE_FIX = pd.Timestamp("2026-05-01")

SERVICE_ASPECTS = ["delivery", "packaging", "value", "customer_service", "returns"]

# ---------------------------------------------------------------------------
# 2. Phrase banks
# ---------------------------------------------------------------------------
FEATURE = {
    "positive": [
        "the {f} is excellent", "{f} is way better than I expected", "really impressed with the {f}",
        "the {f} is top notch", "{f} is superb for the price", "absolutely love the {f}",
        "the {f} has been flawless so far", "{f} is solid and reliable", "the {f} is brilliant",
        "{f} is genuinely the best in this range", "the {f} works perfectly",
    ],
    "negative": [
        "the {f} is terrible", "{f} is far worse than advertised", "really disappointed with the {f}",
        "the {f} stopped working within a week", "{f} is a joke at this price", "the {f} is awful",
        "{f} gave up after just a few days", "the {f} feels cheap and flimsy",
        "the {f} is the worst I have seen", "{f} is completely useless",
    ],
    "neutral": [
        "the {f} is okay", "{f} is average, nothing special", "the {f} is as described",
        "{f} is decent for daily use", "the {f} does the job", "{f} is fine, neither great nor bad",
        "the {f} is about what you would expect", "{f} is standard for this segment",
    ],
}
# Negated phrases: the polarity of the clause is the OPPOSITE of the key word.
NEGATED = {
    "positive": [
        "no complaints at all about the {f}", "the {f} is not bad at all, actually very good",
        "the {f} never disappoints", "can't fault the {f}", "the {f} is not just good, it is great",
    ],
    "negative": [
        "the {f} is not good at all", "the {f} is not worth the money", "{f} is nowhere near what was promised",
        "the {f} doesn't work properly", "can't recommend the {f} to anyone", "the {f} is not even close to good",
    ],
}
# Aspect-specific phrases make the language less generic.
SPECIFIC = {
    "battery": {"positive": ["battery easily lasts two days", "battery backup is amazing"],
                "negative": ["battery drains in a few hours", "battery dies before lunch",
                             "battery drains overnight even when idle"],
                "neutral": ["battery lasts about a day, which is normal"]},
    "sync": {"positive": ["syncs with my phone instantly"],
             "negative": ["keeps disconnecting from the app", "app sync fails every single day"],
             "neutral": ["app sync works most of the time"]},
    "sound": {"positive": ["sound is crisp and the bass is punchy"],
              "negative": ["sound cracks at high volume", "audio is muffled and tinny"],
              "neutral": ["sound is fine for podcasts"]},
    "heat": {"negative": ["it gets dangerously hot while charging"], "positive": ["stays cool even while fast charging"]},
    "assembly": {"negative": ["assembly took two hours and two screws were missing"],
                 "positive": ["assembly took ten minutes, very easy"],
                 "neutral": ["assembly took around half an hour"]},
    "fit": {"negative": ["size runs very small, had to return"], "positive": ["fits true to size"],
            "neutral": ["size is slightly larger than usual"]},
}
SERVICE = {
    "delivery": {
        "positive": ["delivery was super quick", "arrived a day earlier than promised", "delivered on time, no fuss",
                     "the delivery guy was very polite and on time"],
        "negative": ["delivery took almost two weeks", "the parcel was delayed three times",
                     "the delivery person never showed up", "order was stuck in transit for days",
                     "delivery was rescheduled again and again"],
        "neutral": ["delivery took the usual four to five days", "delivered on the expected date",
                    "delivery was neither fast nor slow"],
    },
    "packaging": {
        "positive": ["packaging was neat and secure", "came in really sturdy packaging"],
        "negative": ["box arrived crushed and the seal was broken", "packaging was torn and the item was scratched"],
        "neutral": ["packaging was basic", "came in a plain brown box"],
    },
    "value": {
        "positive": ["great value for money", "worth every penny", "got it at a fantastic price in the sale"],
        "negative": ["total waste of money", "seriously overpriced for what you get", "not worth the money at all"],
        "neutral": ["price is reasonable for what you get", "priced about the same as other brands"],
    },
    "customer_service": {
        "positive": ["customer support solved my issue in minutes", "support team was helpful and quick"],
        "negative": ["customer care keeps giving me copy-paste replies", "raised a complaint 10 days ago and still no response",
                     "support chat closed my ticket without fixing anything"],
        "neutral": ["contacted support to check the warranty, they explained the process",
                    "support said a technician will visit next week"],
    },
    "returns": {
        "positive": ["return and refund were completely hassle-free", "replacement arrived within three days"],
        "negative": ["return request rejected for no reason", "still waiting for my refund after three weeks"],
        "neutral": ["return pickup is scheduled for tomorrow", "requested an exchange for a different size"],
    },
}
SARCASM = [
    "Wow, the {f} lasts a whole two hours. Brilliant engineering.",
    "Great, another {p} that died in a week. Exactly what I paid for.",
    "Love how the {f} stops working the moment the warranty card is filed away. Genius.",
    "Fantastic, ordered a {p} and received an empty box. Amazing service.",
    "Oh perfect, the {f} broke on day two. Just what I needed.",
    "10/10 would definitely enjoy waiting three weeks for a {p} again.",
    "Best {p} ever, if you enjoy restarting it every hour.",
    "Thanks for the {p} that heats up like a stove. Really appreciate it.",
]
WAIT_AND_SEE = [
    "Received the {p} yesterday, will update after a week of use.",
    "Just unboxed the {p}, too early to say anything.",
    "Ordered the {p} today, let's see how it goes.",
    "Got the {p} as a gift, still setting it up.",
    "Using the {p} for two days now, no strong opinion yet.",
    "Does the {p} support fast charging? Thinking of buying one.",
    "Switched to the {p} from my old one, still getting used to it.",
]
OPENERS = ["Bought the {p} last week.", "Using this {p} for a month now.", "Got this {p} during the sale.",
           "Second {p} I have bought from this seller.", "Ordered the {p} for my brother.",
           "Upgraded to the {p} recently.", "Bought this {p} after reading the reviews.", "", "", ""]
CLOSERS = {
    "positive": ["Highly recommended!", "Will buy again.", "Five stars from me.", "Very happy with the purchase.",
                 "Go for it.", "", ""],
    "negative": ["Do not buy.", "Returning it.", "Very disappointed.", "Avoid this one.", "Regret buying it.", "", ""],
    "neutral": ["It's okay for the price.", "Average product overall.", "Let's see how it holds up.",
                "Three stars.", "Nothing more to add.", "", ""],
}
# Twitter-specific decoration
HANDLES = ["@FlipkartSupport", "@Flipkart"]
HASHTAGS = {
    "positive": ["#happycustomer", "#loveit", "#recommended", "#unboxing", "#techreview"],
    "negative": ["#fail", "#worstservice", "#disappointed", "#refundplease", "#scam"],
    "neutral": ["#newpurchase", "#unboxing", "#firstimpressions", "#shopping"],
}
EMOJI = {"positive": ["😍", "🔥", "👍", "❤️", "😊"], "negative": ["😡", "👎", "😤", "🙄", "💔"],
         "neutral": ["🤔", "📦", "🙂"]}
TW_FILLERS = {"positive": ["ngl", "honestly", "tbh", "so far so good,"], "negative": ["seriously", "tbh", "ugh,", "unbelievable,"],
              "neutral": ["fyi", "update:", "so"]}


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def feature_clause(rng, product, aspect, polarity, allow_negation=True):
    """Return (clause, used_negation) describing one product feature."""
    _, feats, _ = PRODUCTS[product]
    f = feats[aspect]
    spec = SPECIFIC.get(aspect, {}).get(polarity, [])
    if allow_negation and polarity in NEGATED and rng.random() < 0.18:
        return rng.choice(NEGATED[polarity]).format(f=f), True
    if spec and rng.random() < 0.35:
        return rng.choice(spec), False
    return rng.choice(FEATURE[polarity]).format(f=f), False


def service_clause(rng, aspect, polarity):
    return rng.choice(SERVICE[aspect][polarity])


def any_clause(rng, product, aspect, polarity):
    if aspect in SERVICE_ASPECTS:
        return service_clause(rng, aspect, polarity), False
    return feature_clause(rng, product, aspect, polarity)


def pick_aspect(rng, product, sentiment, date, source):
    """Choose what the review is mainly about. Encodes the business stories."""
    product_aspects = list(PRODUCTS[product][1].keys())
    p_service = 0.35 if source == "Flipkart" else 0.50
    month = date.month
    # Festive sale (Oct-Nov) and monsoon (Jul-Aug): delivery dominates complaints
    if sentiment == "negative" and (month in (10, 11) or month in (7, 8)) and rng.random() < 0.45:
        return "delivery"
    # PulseFit bug window: battery / sync complaints dominate
    if product == PULSEFIT and sentiment == "negative" and date < FIRMWARE_FIX and rng.random() < 0.75:
        return rng.choice(["battery", "sync"])
    # Twitter complaints are disproportionately about customer care
    if source == "Twitter" and sentiment == "negative" and rng.random() < 0.25:
        return "customer_service"
    if rng.random() < p_service:
        return rng.choice(SERVICE_ASPECTS)
    return rng.choice(product_aspects)


def sentiment_probs(product, source, date):
    """Probability of [positive, neutral, negative] for a review."""
    p = np.array([0.48, 0.30, 0.22]) if source == "Flipkart" else np.array([0.42, 0.13, 0.45])
    tilt = PRODUCTS[product][2]
    p = p + np.array([tilt, 0.0, -tilt])
    if date.month in (10, 11):                     # festive sale: delivery strain
        p = p + np.array([-0.06, 0.0, 0.06])
    if date.month in (7, 8):                       # monsoon: delivery delays
        p = p + np.array([-0.05, 0.0, 0.05])
    if product == PULSEFIT:
        if date < FIRMWARE_FIX:                    # buggy launch
            p = np.array([0.20, 0.15, 0.65]) if source == "Flipkart" else np.array([0.12, 0.06, 0.82])
        else:                                      # after firmware fix
            p = p + np.array([0.12, 0.0, -0.12])
    p = np.clip(p, 0.02, None)
    return p / p.sum()


def rating_for(rng, sentiment, pattern):
    if pattern == "sarcasm":
        return 1
    if pattern == "mixed":
        return int(rng.choice([4, 3], p=[0.75, 0.25])) if sentiment == "positive" else int(rng.choice([2, 3], p=[0.7, 0.3]))
    table = {"positive": ([5, 4, 3], [0.55, 0.38, 0.07]),
             "neutral": ([3, 4, 2], [0.60, 0.20, 0.20]),
             "negative": ([1, 2, 3], [0.55, 0.33, 0.12])}
    vals, probs = table[sentiment]
    return int(rng.choice(vals, p=probs))


# ---------------------------------------------------------------------------
# 4. Review builders
# ---------------------------------------------------------------------------
def build_review(rng, product, sentiment, aspect, source, date):
    """Return (text, pattern)."""
    _, feats, _ = PRODUCTS[product]
    pattern = "plain"
    pf_fixed = product == PULSEFIT and date >= FIRMWARE_FIX

    # --- special constructions -------------------------------------------
    r = rng.random()
    if sentiment == "negative" and r < (0.10 if source == "Twitter" else 0.04):
        f = feats.get(aspect, rng.choice(list(feats.values())))
        body = rng.choice(SARCASM).format(f=f, p=product)
        pattern = "sarcasm"
        return decorate(rng, body, sentiment, source, product, sarcasm=True), pattern
    if sentiment == "neutral" and r < 0.35:
        body = rng.choice(WAIT_AND_SEE).format(p=product)
        return decorate(rng, body, sentiment, source, product, skip_opener=True), "wait_and_see"

    main, neg_used = any_clause(rng, product, aspect, sentiment)
    if neg_used:
        pattern = "negation"

    # Mixed reviews: overall label stays, but a contrasting clause is added
    other_aspect = rng.choice([a for a in list(feats) + SERVICE_ASPECTS if a != aspect])
    if sentiment in ("positive", "negative") and rng.random() < 0.22:
        opp = "negative" if sentiment == "positive" else "positive"
        contra, _ = any_clause(rng, product, other_aspect, opp)
        if sentiment == "positive":
            templ = rng.choice(["Only downside: {c}. Otherwise {m}.", "{M}. Yes, {c}, but overall I am happy.",
                                "{C}, but honestly {m} and that matters more."])
        else:
            templ = rng.choice(["{C}, but {m}.", "Sure, {c}, however {m}.", "{M}, even though {c}."])
        body = templ.format(c=contra, m=main, C=cap(contra), M=cap(main))
        pattern = "mixed"
    elif sentiment == "neutral" and rng.random() < 0.30:
        # balanced pros and cons -> neutral
        good, _ = any_clause(rng, product, aspect, "positive")
        bad, _ = any_clause(rng, product, other_aspect, "negative")
        body = rng.choice(["{G} but {b}. Average overall.", "Pros: {g}. Cons: {b}. Mixed feelings.",
                           "{G}, although {b}, so it balances out."]).format(g=good, b=bad, G=cap(good), B=cap(bad))
        pattern = "mixed"
    else:
        body = cap(main) + "."
        if rng.random() < 0.45:                    # a supporting second clause
            extra_pol = sentiment if rng.random() < 0.7 else "neutral"
            extra, n2 = any_clause(rng, product, other_aspect, extra_pol)
            body += " " + cap(extra) + "."
            if n2 and pattern == "plain":
                pattern = "negation"

    if product == PULSEFIT and aspect in ("battery", "sync"):
        if pf_fixed and sentiment == "positive":
            body = rng.choice(["After the latest firmware update, ", "Since the May update, "]) + body[0].lower() + body[1:]
        elif not pf_fixed and sentiment == "negative" and rng.random() < 0.5:
            body += " Hope they fix this with an update."

    return decorate(rng, body, sentiment, source, product), pattern


def decorate(rng, body, sentiment, source, product, sarcasm=False, skip_opener=False):
    """Wrap the core opinion in platform-specific style."""
    if source == "Flipkart":
        opener = "" if skip_opener else rng.choice(OPENERS).format(p=product)
        closer = "" if sarcasm else rng.choice(CLOSERS[sentiment])
        text = " ".join(x for x in [opener, body, closer] if x)
        # a few typos / missing punctuation, as in real marketplace reviews
        if rng.random() < 0.06:
            text = text.replace("really", "realy").replace("received", "recieved")
        return text
    # Twitter: shorter, informal, handles, hashtags, emojis, links
    parts = []
    if sentiment == "negative" and rng.random() < 0.55:
        parts.append(rng.choice(HANDLES))
    elif rng.random() < 0.15:
        parts.append(rng.choice(HANDLES))
    if rng.random() < 0.3:
        parts.append(rng.choice(TW_FILLERS[sentiment]))
        body = body[0].lower() + body[1:]
    parts.append(body)
    if rng.random() < 0.6:
        tag_pol = "negative" if sarcasm else sentiment
        parts.append(" ".join(rng.sample(HASHTAGS[tag_pol], k=rng.choice([1, 1, 2]))))
    if rng.random() < 0.5:
        emo_pol = "negative" if sarcasm and rng.random() < 0.5 else ("positive" if sarcasm else sentiment)
        parts.append(rng.choice(EMOJI[emo_pol]))
    if rng.random() < 0.12:
        parts.append("https://t.co/" + "".join(rng.choices("abcdefghijkmnpqrstuvwxyz0123456789", k=10)))
    text = " ".join(parts)
    if rng.random() < 0.35:
        text = text.lower()
    if rng.random() < 0.08:
        text = text.replace("so ", "sooo ").replace("very", "veryyy")
    return text


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def month_starts():
    return pd.date_range("2025-10-01", "2026-09-01", freq="MS")


def generate(n=N_REVIEWS, seed=SEED) -> pd.DataFrame:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    months = month_starts()
    # review volume per month: festive sale peak in Oct/Nov
    vol = np.array([1.6, 1.45, 1.0, 0.9, 1.0, 1.05, 0.9, 0.9, 0.9, 1.0, 1.0, 0.9])
    vol = vol / vol.sum()

    rows, seen = [], set()
    while len(rows) < n:
        m = months[nrng.choice(len(months), p=vol)]
        date = m + pd.Timedelta(days=int(nrng.integers(0, m.days_in_month)))
        # PulseFit is not on sale before launch; afterwards launch buzz gives it
        # roughly double the review volume of a typical product
        weights = [0 if (p == PULSEFIT and date < PULSEFIT_LAUNCH) else (2.0 if p == PULSEFIT else 1.0)
                   for p in PRODUCTS]
        product = rng.choices(list(PRODUCTS), weights=weights)[0]
        source = "Twitter" if rng.random() < TWITTER_SHARE else "Flipkart"
        sentiment = ["positive", "neutral", "negative"][nrng.choice(3, p=sentiment_probs(product, source, date))]
        aspect = pick_aspect(rng, product, sentiment, date, source)
        text, pattern = build_review(rng, product, sentiment, aspect, source, date)
        if text in seen:                            # keep texts unique
            continue
        seen.add(text)
        rating = rating_for(nrng, sentiment, pattern)
        label = sentiment
        if rng.random() < LABEL_NOISE:              # annotator disagreement
            label = {"positive": "neutral", "negative": "neutral",
                     "neutral": rng.choice(["positive", "negative"])}[sentiment]
            pattern = "noisy_label"
        rows.append(dict(review_text=text, product=product, category=PRODUCTS[product][0],
                         aspect=aspect, sentiment=label, source=source, date=date.date().isoformat(),
                         rating=rating, pattern=pattern))

    df = pd.DataFrame(rows).sort_values("date", kind="stable").reset_index(drop=True)
    df.insert(0, "review_id", np.arange(1, len(df) + 1))
    return df


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(DATA_DIR / "synthetic_reviews.csv", index=False)
    df.to_excel(DATA_DIR / "synthetic_reviews.xlsx", index=False)
    print(df.shape)
    print(df["sentiment"].value_counts(normalize=True).round(3))
    print(df["source"].value_counts())
    print(df["pattern"].value_counts())
