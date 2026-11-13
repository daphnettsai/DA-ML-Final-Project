"""recommender tests: synthetic restaurants, no Yelp."""

import pandas as pd
import pytest

from restaurant_alts.recommender import Filters, effective_max_reviews, explain, rank, recommend

LAT, LON = 40.7300, -74.0000
KM = 0.009  # ~1 km of latitude

COLUMNS = ["restaurant_id", "name", "categories", "lat", "lon", "rating", "review_count",
           "price_level", "data_fetched_at"]

REF = {"restaurant_id": "ref", "name": "Reference Place", "categories": "Italian, Pizza",
       "lat": LAT, "lon": LON, "rating": 4.6, "review_count": 1000, "price_level": 2}

NO_FILTERS = Filters(min_rating=None, min_reviews=None, review_ratio=None)


def row(rid, cats="Italian, Pizza", km=0.5, rating=4.5, reviews=100, price=2, name=None):
    lat = None if km is None else LAT + km * KM
    return (rid, name or f"Place {rid}", cats, lat, None if km is None else LON,
            rating, reviews, price, "2026-01-15T12:00:00+00:00")


def pool(*rows):
    return pd.DataFrame(list(rows), columns=COLUMNS)


def ids(result):
    return [r["restaurant_id"] for r in result.recommendations]


# ---------------------------------------------------------------- basics

def test_reference_never_returned():
    p = pool(row("ref"), row("a"), row("b"))
    for f in (NO_FILTERS, Filters()):
        assert "ref" not in ids(recommend(REF, p, f))
        assert "ref" not in ids(recommend(REF, p, f, yelp_available=False))


def test_threshold_gate_and_no_candidate_above():
    p = pool(row("thai", cats="Thai"), row("bars", cats="Bars"))
    r = recommend(REF, p, NO_FILTERS)
    assert r.recommendations == []
    assert "similarity" in r.empty_reason


def test_reference_without_usable_categories():
    ref = {**REF, "categories": "Restaurants, Food"}
    r = recommend(ref, pool(row("a")), NO_FILTERS)
    assert r.recommendations == [] and "no usable cuisine" in r.empty_reason


def test_top_10_returned_from_stage1_30():
    p = pool(*[row(f"r{i:02d}") for i in range(40)])
    r = recommend(REF, p, NO_FILTERS)
    assert r.stage1_count == 30 and len(r.recommendations) == 10


def test_required_cuisine():
    p = pool(row("it", cats="Italian"), row("pz", cats="Pizza"))
    r = recommend(REF, p, Filters(required_cuisine="italian", min_rating=None,
                                  min_reviews=None, review_ratio=None))
    assert ids(r) == ["it"]


# ---------------------------------------------------------------- review limits

def test_effective_max_reviews():
    assert effective_max_reviews(1000, Filters(review_ratio=0.5))[0] == 500
    assert effective_max_reviews(1000, Filters(review_ratio=0.25, absolute_max_reviews=100))[0] == 100
    assert effective_max_reviews(1001, Filters(review_ratio=0.5))[0] == 500  # int() floors
    assert effective_max_reviews(None, Filters(review_ratio=0.5)) == (
        None, ["Relative review limit disabled: the reference's review count is unknown."])


def test_review_limits_are_inclusive():
    p = pool(row("min", reviews=20), row("max", reviews=500), row("lo", reviews=19),
             row("hi", reviews=501))
    r = recommend(REF, p, Filters(min_rating=None, min_reviews=20, review_ratio=0.5))
    assert set(ids(r)) == {"min", "max"}
    assert r.applied_filters["max_reviews"] == 500


def test_missing_reference_reviews_disables_relative_filter_visibly():
    ref = {**REF, "review_count": None}
    p = pool(row("big", reviews=5000))
    r = recommend(ref, p, Filters(min_rating=None, min_reviews=20, review_ratio=0.5))
    assert ids(r) == ["big"]
    assert any("Relative review limit disabled" in n for n in r.notes)


def test_conflicting_review_thresholds():
    ref = {**REF, "review_count": 30}  # 50% -> 15, below min 20
    r = recommend(ref, pool(row("a", reviews=18)), Filters(min_rating=None, min_reviews=20))
    assert r.recommendations == []
    assert "conflict" in r.empty_reason and "15" in r.empty_reason
    assert r.relax_suggestions


def test_unknown_rating_or_reviews_fail_minimums():
    p = pool(row("norating", rating=None), row("noreviews", reviews=None), row("ok"))
    r = recommend(REF, p, Filters(min_rating=4.0, min_reviews=20, review_ratio=None))
    assert ids(r) == ["ok"]


def test_empty_state_names_filters_to_relax():
    p = pool(row("a", rating=3.0), row("b", rating=3.5))
    r = recommend(REF, p, Filters(min_rating=4.0, min_reviews=None, review_ratio=None))
    assert r.recommendations == []
    assert any("minimum rating" in s for s in r.relax_suggestions)


def test_fallback_disables_rating_and_review_filters_with_note():
    p = pool(row("a", rating=None, reviews=None))
    r = recommend(REF, p, Filters(), yelp_available=False)
    assert ids(r) == ["a"]
    assert any("disabled because Yelp data is unavailable" in n for n in r.notes)
    assert "min_rating" not in r.applied_filters


# ---------------------------------------------------------------- price and distance filters

def test_unknown_price_fails_price_cap():
    p = pool(row("cheap", price=1), row("unknown", price=None), row("pricey", price=3))
    r = recommend(REF, p, Filters(max_price_level=2, min_rating=None, min_reviews=None,
                                  review_ratio=None))
    assert ids(r) == ["cheap"]


def test_distance_filter_excludes_far_and_unknown():
    p = pool(row("near", km=0.5), row("far", km=4), row("nowhere", km=None))
    r = recommend(REF, p, Filters(max_distance_km=2, min_rating=None, min_reviews=None,
                                  review_ratio=None))
    assert ids(r) == ["near"]


def test_distance_filter_with_unknown_reference_location_is_not_dropped():
    ref = {**REF, "lat": None, "lon": None}
    r = recommend(ref, pool(row("a")), Filters(max_distance_km=2, min_rating=None,
                                                min_reviews=None, review_ratio=None))
    assert r.recommendations == []
    assert "location is unknown" in r.empty_reason


# ---------------------------------------------------------------- tiered ranking

def test_higher_cuisine_band_always_first():
    p = pool(row("exact_far", cats="Italian, Pizza", km=8, price=4, rating=3.0),
             row("partial_near", cats="Italian, Bars", km=0.1, price=2, rating=5.0))
    assert ids(recommend(REF, p, NO_FILTERS))[0] == "exact_far"


def test_within_band_distance_then_price():
    p = pool(row("d2_p0", km=2.0, price=2),
             row("d0_p2", km=0.5, price=4),
             row("d0_p0", km=0.8, price=2),
             row("d3", km=10, price=2))
    assert ids(recommend(REF, p, NO_FILTERS)) == ["d0_p0", "d0_p2", "d2_p0", "d3"]


def test_rating_then_exact_distance_then_name():
    p = pool(row("b", km=0.5, rating=4.2, name="Bravo"),
             row("a", km=0.5, rating=4.8, name="Alpha"),
             row("c", km=0.2, rating=4.2, name="Charlie"),
             row("d", km=0.2, rating=4.2, name="Able"))
    assert ids(recommend(REF, p, NO_FILTERS)) == ["a", "d", "c", "b"]


def test_unknown_values_never_rank_above_known_within_band():
    # Unknown distance ranks below even the farthest known band.
    p = pool(row("nodist", km=None, rating=5.0), row("far", km=9, rating=3.0))
    assert ids(recommend(REF, p, NO_FILTERS)) == ["far", "nodist"]
    # Same distance band: unknown price ranks below the largest known price gap.
    p = pool(row("noprice", km=0.5, price=None, rating=5.0), row("gap2", km=0.5, price=4, rating=3.0))
    assert ids(recommend(REF, p, NO_FILTERS)) == ["gap2", "noprice"]


def test_missing_rating_sorts_after_rated_on_tie():
    p = pool(row("rated", rating=3.0), row("unrated", rating=None))
    assert ids(recommend(REF, p, NO_FILTERS)) == ["rated", "unrated"]


def test_reference_missing_price_makes_price_tier_inert():
    ref = {**REF, "price_level": None}
    p = pool(row("p4", price=4, rating=4.9, km=0.5), row("p2", price=2, rating=4.1, km=0.5))
    r = recommend(ref, p, NO_FILTERS)
    assert ids(r) == ["p4", "p2"]  # decided by rating, not price
    assert all(x["price_gap"] == 99 for x in r.recommendations)
    assert any("Price tier skipped" in n for n in r.notes)
    assert r.recommendations[0]["tiers_skipped"] == [
        {"tier": "price", "reason": "reference price unknown"}]


def test_reference_missing_coordinates_makes_distance_tier_inert():
    ref = {**REF, "lat": None, "lon": None}
    p = pool(row("far_good", km=9, rating=4.9), row("near_ok", km=0.1, rating=4.0))
    r = recommend(ref, p, NO_FILTERS)
    assert ids(r) == ["far_good", "near_ok"]
    assert all(x["distance_band"] == 99 and x["distance_km"] is None for x in r.recommendations)
    assert any("Distance tier skipped" in n for n in r.notes)


def test_tier_disclosure_fields():
    p = pool(row("full"), row("nodist", km=None), row("noprice", price=None))
    recs = {x["restaurant_id"]: x for x in recommend(REF, p, NO_FILTERS).recommendations}
    assert recs["full"]["tiers_used"] == ["cuisine", "distance", "price"]
    assert recs["full"]["tiers_skipped"] == []
    assert recs["nodist"]["tiers_skipped"] == [
        {"tier": "distance", "reason": "candidate location unknown"}]
    assert recs["noprice"]["tiers_used"] == ["cuisine", "distance"]


# ---------------------------------------------------------------- output and explanations

def test_output_schema():
    rec = recommend(REF, pool(row("a", reviews=150)), NO_FILTERS).recommendations[0]
    assert set(rec) == {
        "restaurant_id", "name", "shared_categories", "similarity_score", "label", "rating",
        "review_count", "review_ratio", "price_level", "distance_km", "distance_band",
        "price_gap", "data_fetched_at", "tiers_used", "tiers_skipped", "explanation"}
    assert rec["review_ratio"] == 0.15
    assert rec["label"] == "very similar"
    assert rec["shared_categories"] == ["Italian", "Pizza"]


def test_explanation_templates():
    assert explain(["Italian", "Pizza"], 0.72, 1.234, 2, 2, None) == (
        "Shares the Italian and Pizza categories (cuisine similarity 0.72); 1.2 km away; "
        "same price level.")
    assert explain(["Italian"], 0.5, None, None, 2, 0.15) == (
        "Shares the Italian category (cuisine similarity 0.50); has 15% as many reviews as "
        "your reference; price information is unavailable.")
    assert explain(["Thai"], 0.9, 0.4, 3, 1, None).endswith("2 price levels higher.")


def test_rank_is_stable_for_identical_rows():
    p = pool(row("x", name="Same"), row("y", name="Same"))
    r = recommend(REF, p, NO_FILTERS)
    assert ids(r) == ["x", "y"]


def test_invalid_filter_values_raise():
    with pytest.raises(ValueError):
        recommend(REF, pool(row("a")), Filters(max_price_level=5))
    with pytest.raises(ValueError):
        recommend(REF, pool(row("a")), Filters(min_rating=6))
