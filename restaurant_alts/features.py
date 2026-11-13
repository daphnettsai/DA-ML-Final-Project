"""Feature helpers: category tokens, TF-IDF cuisine similarity, distance, ranking bands.

Similarity uses categories only. Rating, review count, price and distance are never part
of the score (spec section 6).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

GENERIC_TAGS = frozenset({"restaurants", "restaurant", "food"})
SIMILARITY_THRESHOLD = 0.30
UNKNOWN = 99  # band / gap value for unknown distance or price: always ranks last


# ---------------------------------------------------------------- categories

def _split(categories) -> list[str]:
    if categories is None or (isinstance(categories, float) and math.isnan(categories)):
        return []
    if categories is pd.NA:
        return []
    if isinstance(categories, str):
        return [c.strip() for c in categories.split(",")]
    return [str(c).strip() for c in categories]


def to_token(title: str) -> str:
    return "_".join(title.lower().split())


def category_tokens(categories) -> list[str]:
    """'Italian, Pizza, Restaurants' -> ['italian', 'pizza']. Order kept, duplicates dropped."""
    tokens = (to_token(c) for c in _split(categories) if c)
    return list(dict.fromkeys(t for t in tokens if t and t not in GENERIC_TAGS))


def shared_categories(reference, candidate) -> list[str]:
    """Candidate's category titles that the reference also has (display form)."""
    ref = set(category_tokens(reference))
    out = []
    for title in _split(candidate):
        if title and to_token(title) in ref and to_token(title) not in GENERIC_TAGS:
            if title not in out:
                out.append(title)
    return out


def has_category(categories, required: str) -> bool:
    return to_token(required.strip()) in category_tokens(categories)


# ---------------------------------------------------------------- similarity

def fit_vectorizer(pool_categories: Iterable) -> TfidfVectorizer | None:
    """Fit TF-IDF on the candidate pool. None when the pool has no usable categories."""
    vec = TfidfVectorizer(analyzer=category_tokens)
    try:
        vec.fit(list(pool_categories))
    except ValueError:  # empty vocabulary
        return None
    return vec


def cuisine_similarity(reference_categories, pool_categories: Sequence) -> np.ndarray:
    """Cosine similarity (0-1) of the reference to each pool item.

    The vectorizer is fit on the pool and the reference is transformed with the SAME
    vectorizer, so a reference outside the pool works and its unseen categories are ignored.
    """
    pool = list(pool_categories)
    if not pool:
        return np.zeros(0)
    vec = fit_vectorizer(pool)
    if vec is None or not category_tokens(reference_categories):
        return np.zeros(len(pool))
    ref_vec = vec.transform([reference_categories])
    if ref_vec.nnz == 0:  # none of the reference's categories appear in the pool
        return np.zeros(len(pool))
    scores = cosine_similarity(ref_vec, vec.transform(pool)).ravel()
    return np.clip(scores, 0.0, 1.0)


def similarity_label(score: float) -> str | None:
    """Label from cuisine similarity only."""
    hundredths = int(round(score * 100))
    if hundredths >= 80:
        return "very similar"
    if hundredths >= 50:
        return "similar"
    if hundredths >= 30:
        return "loosely similar"
    return None


def cuisine_band(score: float) -> int:
    """-floor(score * 10), computed on the 2-decimal score so 0.80 is band 8, not 7."""
    return -(int(round(score * 100)) // 10)


# ---------------------------------------------------------------- distance and price

def is_missing(x) -> bool:
    """None, NaN, pd.NA and NaT all count as missing."""
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except (TypeError, ValueError):  # list-like values are not "missing"
        return False


_missing = is_missing


def haversine_km(lat1, lon1, lat2, lon2) -> float | None:
    """Great-circle distance in km, or None when any coordinate is missing."""
    if any(_missing(v) for v in (lat1, lon1, lat2, lon2)):
        return None
    lat1, lon1, lat2, lon2 = map(math.radians, map(float, (lat1, lon1, lat2, lon2)))
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def distance_band(km) -> int:
    if _missing(km):
        return UNKNOWN
    if km <= 1:
        return 0
    if km <= 3:
        return 1
    if km <= 5:
        return 2
    return 3


def price_gap(candidate_level, reference_level) -> int:
    if _missing(candidate_level) or _missing(reference_level):
        return UNKNOWN
    return abs(int(candidate_level) - int(reference_level))
