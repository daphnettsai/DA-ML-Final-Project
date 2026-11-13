"""features tests (synthetic categories only)."""

import math

import numpy as np
import pandas as pd
import pytest

from restaurant_alts import features as ft


def test_category_tokens_drop_generic_and_normalize():
    assert ft.category_tokens("Italian, Restaurants, New American, Food, italian") == [
        "italian", "new_american"]
    assert ft.category_tokens(None) == []
    assert ft.category_tokens(float("nan")) == []
    assert ft.category_tokens(["Thai", " Noodles "]) == ["thai", "noodles"]


def test_shared_categories_display_titles():
    assert ft.shared_categories("Italian, Pizza", "Pizza, Italian, Bars") == ["Pizza", "Italian"]
    assert ft.shared_categories("Thai", "Italian") == []


def test_has_category_exact_token():
    assert ft.has_category("Italian, Pizza", "italian")
    assert ft.has_category("New American", "New American")
    assert not ft.has_category("Italian", "Ital")


def test_identical_categories_score_one_and_disjoint_zero():
    s = ft.cuisine_similarity("Italian, Pizza", ["Italian, Pizza", "Thai", "Italian"])
    assert s[0] == pytest.approx(1.0)
    assert s[1] == 0
    assert 0 < s[2] < 1


def test_reference_outside_pool_and_unseen_categories_ignored():
    # 'georgian' never appears in the pool: ignored, 'italian' still counts.
    s = ft.cuisine_similarity("Georgian, Italian", ["Italian", "Thai"])
    assert s[0] == pytest.approx(1.0)
    assert s[1] == 0


def test_no_usable_categories_gives_zeros():
    assert not ft.cuisine_similarity("Restaurants, Food", ["Italian", "Thai"]).any()
    assert not ft.cuisine_similarity("Italian", ["Restaurants", None]).any()
    assert not ft.cuisine_similarity("Georgian", ["Italian", "Thai"]).any()
    assert ft.cuisine_similarity("Italian", []).size == 0


@pytest.mark.parametrize("score, label, band", [
    (0.80, "very similar", -8),
    (0.7999, "very similar", -8),   # rounds to 0.80
    (0.79, "similar", -7),
    (0.50, "similar", -5),
    (0.49, "loosely similar", -4),
    (0.30, "loosely similar", -3),
    (0.29, None, -2),
    (1.0, "very similar", -10),
])
def test_label_and_band_boundaries(score, label, band):
    assert ft.similarity_label(score) == label
    assert ft.cuisine_band(score) == band


def test_haversine_known_distance_and_missing():
    # 0.01 degree of latitude is about 1.11 km
    assert ft.haversine_km(40.0, -74.0, 40.01, -74.0) == pytest.approx(1.11, abs=0.01)
    assert ft.haversine_km(40.0, -74.0, None, -74.0) is None
    assert ft.haversine_km(40.0, math.nan, 40.0, -74.0) is None
    assert ft.haversine_km(40.0, -74.0, pd.NA, -74.0) is None


@pytest.mark.parametrize("km, band", [
    (0.0, 0), (1.0, 0), (1.01, 1), (3.0, 1), (5.0, 2), (5.01, 3), (None, 99), (np.nan, 99),
])
def test_distance_band(km, band):
    assert ft.distance_band(km) == band


def test_price_gap():
    assert ft.price_gap(2, 2) == 0
    assert ft.price_gap(4, 1) == 3
    assert ft.price_gap(None, 2) == 99
    assert ft.price_gap(2, None) == 99
    assert ft.price_gap(pd.NA, 2) == 99
