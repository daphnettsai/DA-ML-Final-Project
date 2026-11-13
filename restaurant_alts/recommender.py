"""Two-stage recommender with strict filters, tiered ranking and template explanations.

Stage 1 (local): exclude the reference, apply cuisine and distance filters, keep cuisine
similarity >= 0.30, take the top 30 by similarity.
Stage 2 (after Yelp enrichment, done by the caller): apply rating, review and price
filters, rank with the tiered key, return the top 10.

Inputs are plain DataFrames / mappings with these columns:
    restaurant_id, name, categories, lat, lon, rating, review_count, price_level,
    data_fetched_at (optional)
Filters are never relaxed or changed silently: every disabled filter or conflict is
reported in ``notes``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping

import numpy as np
import pandas as pd

from restaurant_alts import features as ft
from restaurant_alts.name_matching import normalize_name

STAGE1_LIMIT = 30
FINAL_LIMIT = 10
SCORE_NOTE = "Score reflects category overlap only, not menu, atmosphere, or quality."

TIER_CUISINE, TIER_DISTANCE, TIER_PRICE = "cuisine", "distance", "price"


@dataclass
class Filters:
    required_cuisine: str | None = None
    min_rating: float | None = 4.0
    min_reviews: int | None = 20
    review_ratio: float | None = 0.5          # max reviews relative to reference
    absolute_max_reviews: int | None = None
    max_price_level: int | None = None        # 1-4
    max_distance_km: float | None = None

    def validate(self) -> None:
        if self.min_rating is not None and not 0 <= self.min_rating <= 5:
            raise ValueError("Minimum rating must be between 0 and 5.")
        for name in ("min_reviews", "absolute_max_reviews"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative.")
        if self.review_ratio is not None and self.review_ratio <= 0:
            raise ValueError("Review ratio must be above 0.")
        if self.max_price_level is not None and self.max_price_level not in (1, 2, 3, 4):
            raise ValueError("Price level must be 1, 2, 3, or 4.")
        if self.max_distance_km is not None and self.max_distance_km <= 0:
            raise ValueError("Maximum distance must be above 0.")


@dataclass
class Result:
    recommendations: list[dict] = field(default_factory=list)
    applied_filters: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)       # disabled filters, conflicts, tiers
    empty_reason: str | None = None
    relax_suggestions: list[str] = field(default_factory=list)
    stage1_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- helpers

def _get(row: Mapping, key):
    value = row.get(key) if hasattr(row, "get") else None
    return None if ft.is_missing(value) else value


def effective_max_reviews(reference_reviews, filters: Filters) -> tuple[int | None, list[str]]:
    """min(absolute_max, int(reference_reviews * ratio)). Missing pieces are reported."""
    notes = []
    relative = None
    if filters.review_ratio is not None:
        if ft.is_missing(reference_reviews):
            notes.append("Relative review limit disabled: the reference's review count is unknown.")
        else:
            relative = int(reference_reviews * filters.review_ratio)
    caps = [c for c in (relative, filters.absolute_max_reviews) if c is not None]
    return (min(caps) if caps else None), notes


NUMERIC_COLUMNS = ("lat", "lon", "rating", "review_count", "price_level")


def not_reference(reference: Mapping, df: pd.DataFrame) -> pd.Series:
    """Rows that are NOT the reference: different id, different Yelp ID, and not the same
    normalized name in the same ZIP (catches a Yelp-only listing of the same restaurant)."""
    keep = df["restaurant_id"].ne(_get(reference, "restaurant_id"))
    ref_yelp = _get(reference, "yelp_id")
    if ref_yelp is not None and "yelp_id" in df.columns:
        keep &= df["yelp_id"].ne(ref_yelp) | df["yelp_id"].isna()
    ref_name, ref_zip = normalize_name(_get(reference, "name")), _get(reference, "zip")
    if ref_name and ref_zip is not None and "zip" in df.columns:
        same = df["name"].map(normalize_name).eq(ref_name) & df["zip"].astype(str).eq(str(ref_zip))
        keep &= ~same
    return keep


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns as floats with NaN for missing (None/pd.NA/'' all become NaN)."""
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    return df


def _empty(result: Result, reason: str, suggestions: list[str] | None = None) -> Result:
    result.recommendations = []
    result.empty_reason = reason
    result.relax_suggestions = suggestions or []
    return result


# ---------------------------------------------------------------- stage 1

def stage1(reference: Mapping, pool: pd.DataFrame, filters: Filters) -> tuple[pd.DataFrame, Result]:
    """Local retrieval. Returns (top candidates, partial result with notes / empty state)."""
    filters.validate()
    result = Result()
    ref_id = _get(reference, "restaurant_id")
    cands = _coerce(pool[not_reference(reference, pool)])  # the reference is never returned

    if not ft.category_tokens(_get(reference, "categories")):
        return cands.iloc[0:0], _empty(
            result, "The reference restaurant has no usable cuisine categories, "
                    "so similarity can't be computed.")

    # Cuisine similarity: fit on the candidate pool, transform the reference.
    cands["similarity"] = ft.cuisine_similarity(_get(reference, "categories"), cands["categories"].tolist())

    # Distance (used by the filter and later ranking).
    ref_lat, ref_lon = _get(reference, "lat"), _get(reference, "lon")
    cands["distance_km"] = [
        ft.haversine_km(ref_lat, ref_lon, la, lo) for la, lo in zip(cands["lat"], cands["lon"])
    ]

    excluded: dict[str, int] = {}

    if filters.required_cuisine:
        keep = cands["categories"].map(lambda c: ft.has_category(c, filters.required_cuisine))
        excluded["required cuisine"] = int((~keep).sum())
        cands = cands[keep]
        result.applied_filters["required_cuisine"] = filters.required_cuisine

    if filters.max_distance_km is not None:
        if ref_lat is None or ref_lon is None:
            return cands.iloc[0:0], _empty(
                result, "The maximum-distance filter can't be applied because the reference "
                        "restaurant's location is unknown.",
                ["Remove the maximum-distance filter."])
        # Unknown distance is not near: it fails the filter.
        keep = cands["distance_km"].map(lambda d: d is not None and d <= filters.max_distance_km)
        excluded["maximum distance"] = int((~keep).sum())
        cands = cands[keep]
        result.applied_filters["max_distance_km"] = filters.max_distance_km

    keep = cands["similarity"].round(2).ge(ft.SIMILARITY_THRESHOLD)
    below = int((~keep).sum())
    cands = cands[keep]

    cands = cands.sort_values(["similarity", "name"], ascending=[False, True]).head(STAGE1_LIMIT)
    result.stage1_count = len(cands)

    if cands.empty:
        if below and not any(excluded.values()):
            return cands, _empty(result, "No restaurant shares enough cuisine categories "
                                         f"(similarity ≥ {ft.SIMILARITY_THRESHOLD:.2f}) with the reference.")
        return cands, _empty(result, "No restaurant passes the cuisine and distance filters.",
                             _suggest(excluded))
    return cands, result


# ---------------------------------------------------------------- stage 2

def stage2(
    reference: Mapping,
    candidates: pd.DataFrame,
    filters: Filters,
    result: Result,
    yelp_available: bool = True,
) -> Result:
    """Apply rating/review/price filters to enriched candidates, rank, explain, top 10."""
    if result.empty_reason:
        return result
    ref_id = _get(reference, "restaurant_id")
    cands = _coerce(candidates[not_reference(reference, candidates)])
    excluded: dict[str, int] = {}
    ref_reviews = _get(reference, "review_count")

    if not yelp_available:
        result.notes.append("Rating and review filters are disabled because Yelp data is "
                            "unavailable; results are ranked from local data only.")
    else:
        if filters.min_rating is not None:
            keep = cands["rating"].ge(filters.min_rating).fillna(False).astype(bool)  # unknown fails
            excluded["minimum rating"] = int((~keep).sum())
            cands = cands[keep]
            result.applied_filters["min_rating"] = filters.min_rating

        max_reviews, notes = effective_max_reviews(ref_reviews, filters)
        result.notes.extend(notes)
        if (filters.min_reviews is not None and max_reviews is not None
                and max_reviews < filters.min_reviews):
            return _empty(
                result,
                f"The review limits conflict: the maximum ({max_reviews} reviews"
                + (f", {filters.review_ratio:.0%} of the reference's {int(ref_reviews)}"
                   if ref_reviews is not None and filters.review_ratio else "")
                + f") is below the minimum ({filters.min_reviews}).",
                ["Lower the minimum reviews.", "Raise the relative review limit."])

        if filters.min_reviews is not None:
            keep = cands["review_count"].ge(filters.min_reviews).fillna(False).astype(bool)
            excluded["minimum reviews"] = int((~keep).sum())
            cands = cands[keep]
            result.applied_filters["min_reviews"] = filters.min_reviews
        if max_reviews is not None:
            keep = cands["review_count"].le(max_reviews).fillna(False).astype(bool)
            excluded["maximum reviews"] = int((~keep).sum())
            cands = cands[keep]
            result.applied_filters["max_reviews"] = max_reviews

    if filters.max_price_level is not None:
        # Unknown price is not cheap: it fails a price cap.
        keep = cands["price_level"].le(filters.max_price_level).fillna(False).astype(bool)
        excluded["price cap"] = int((~keep).sum())
        cands = cands[keep]
        result.applied_filters["max_price_level"] = filters.max_price_level

    if cands.empty:
        return _empty(result, "No restaurants meet all of your filters.", _suggest(excluded))

    ranked = rank(reference, cands)
    result.notes.extend(_tier_notes(reference))
    result.recommendations = [
        _output(reference, row) for _, row in ranked.head(FINAL_LIMIT).iterrows()
    ]
    return result


def recommend(
    reference: Mapping,
    pool: pd.DataFrame,
    filters: Filters | None = None,
    yelp_available: bool = True,
) -> Result:
    """Both stages on data already in hand (used for local-only mode and tests)."""
    filters = filters or Filters()
    cands, result = stage1(reference, pool, filters)
    return stage2(reference, cands, filters, result, yelp_available)


# ---------------------------------------------------------------- ranking

def rank(reference: Mapping, cands: pd.DataFrame) -> pd.DataFrame:
    """Tiered sort: cuisine band, distance band, price gap, -rating, distance, name."""
    cands = cands.copy()
    ref_has_coords = _get(reference, "lat") is not None and _get(reference, "lon") is not None
    ref_price = _get(reference, "price_level")

    cands["cuisine_band"] = cands["similarity"].map(ft.cuisine_band)
    cands["distance_band"] = (cands["distance_km"].map(ft.distance_band)
                              if ref_has_coords else ft.UNKNOWN)
    cands["price_gap"] = cands["price_level"].map(lambda p: ft.price_gap(p, ref_price))
    cands["_rating_missing"] = cands["rating"].isna()
    cands["_neg_rating"] = -cands["rating"].fillna(0)
    cands["_dist_missing"] = cands["distance_km"].isna()
    cands["_dist"] = cands["distance_km"].astype(float).fillna(np.inf)
    cands["_name"] = cands["name"].str.casefold()
    return cands.sort_values(
        ["cuisine_band", "distance_band", "price_gap", "_rating_missing", "_neg_rating",
         "_dist_missing", "_dist", "_name"],
        kind="mergesort",
    )


def _tier_notes(reference: Mapping) -> list[str]:
    notes = []
    if _get(reference, "lat") is None or _get(reference, "lon") is None:
        notes.append("Distance tier skipped: the reference restaurant's location is unknown.")
    if _get(reference, "price_level") is None:
        notes.append("Price tier skipped: the reference restaurant's price level is unknown.")
    return notes


# ---------------------------------------------------------------- output

def _output(reference: Mapping, row: pd.Series) -> dict:
    ref_reviews = _get(reference, "review_count")
    ref_price = _get(reference, "price_level")
    rating = _get(row, "rating")
    reviews = _get(row, "review_count")
    price = _get(row, "price_level")
    dist = _get(row, "distance_km")
    score = round(float(row["similarity"]), 2)
    shared = ft.shared_categories(_get(reference, "categories"), _get(row, "categories"))

    ratio = (reviews / ref_reviews) if (reviews is not None and ref_reviews) else None

    used, skipped = [TIER_CUISINE], []
    if _get(reference, "lat") is None or _get(reference, "lon") is None:
        skipped.append({"tier": TIER_DISTANCE, "reason": "reference location unknown"})
    elif dist is None:
        skipped.append({"tier": TIER_DISTANCE, "reason": "candidate location unknown"})
    else:
        used.append(TIER_DISTANCE)
    if ref_price is None:
        skipped.append({"tier": TIER_PRICE, "reason": "reference price unknown"})
    elif price is None:
        skipped.append({"tier": TIER_PRICE, "reason": "candidate price unknown"})
    else:
        used.append(TIER_PRICE)

    return {
        "restaurant_id": row["restaurant_id"],
        "name": row["name"],
        "shared_categories": shared,
        "similarity_score": score,
        "label": ft.similarity_label(score),
        "rating": rating,
        "review_count": None if reviews is None else int(reviews),
        "review_ratio": None if ratio is None else round(ratio, 2),
        "price_level": None if price is None else int(price),
        "distance_km": None if dist is None else round(float(dist), 1),
        "distance_band": int(row["distance_band"]),
        "price_gap": int(row["price_gap"]),
        "data_fetched_at": _get(row, "data_fetched_at"),
        "tiers_used": used,
        "tiers_skipped": skipped,
        "explanation": explain(shared, score, dist, price, ref_price, ratio),
    }


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def explain(shared, score, distance_km, price, ref_price, review_ratio) -> str:
    """Template explanation. Distance and review clauses appear only when known."""
    if shared:
        noun = "category" if len(shared) == 1 else "categories"
        parts = [f"Shares the {_join(shared)} {noun} (cuisine similarity {score:.2f})"]
    else:
        parts = [f"Cuisine similarity {score:.2f}"]
    if distance_km is not None:
        parts.append(f"{distance_km:.1f} km away")
    if review_ratio is not None:
        parts.append(f"has {review_ratio:.0%} as many reviews as your reference")
    if price is None or ref_price is None:
        parts.append("price information is unavailable")
    else:
        gap = int(price) - int(ref_price)
        if gap == 0:
            parts.append("same price level")
        else:
            steps = "level" if abs(gap) == 1 else "levels"
            parts.append(f"{abs(gap)} price {steps} {'higher' if gap > 0 else 'lower'}")
    return "; ".join(parts) + "."


def _suggest(excluded: dict[str, int]) -> list[str]:
    """Name the filters that removed candidates, biggest first."""
    hits = sorted(((n, k) for k, n in excluded.items() if n), reverse=True)
    return [f"Relax the {k} filter (it excluded {n})." for n, k in hits]
