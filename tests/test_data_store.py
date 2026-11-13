"""data_store tests. All rows are synthetic; no real NYC or Yelp data."""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from restaurant_alts import data_store as ds

NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def conn():
    c = ds.connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def nyc_csv(tmp_path):
    path = tmp_path / "nyc.csv"
    path.write_text(
        "CAMIS,DBA,ADDRESS,ZIPCODE,Latitude,Longitude,CUISINE_DESCRIPTION\n"
        "1001,Test Trattoria,1 Fake St,10001,40.70,-74.00,Italian\n"
        "1001,Test Trattoria,1 Fake St,10001,40.70,-74.00,Italian\n"  # repeat inspection row
        "1002,Sample Noodle Bar,2 Fake Ave,10002,0,0,Chinese\n"        # 0 = no coordinates
        "1003,Example Diner,,10003,,,\n"
    )
    return path


def test_import_directory_handles_nyc_columns_and_duplicates(conn, nyc_csv):
    assert ds.import_directory_csv(conn, nyc_csv, source="synthetic", imported_at=NOW) == 3
    df = ds.load_directory(conn).set_index("nyc_id")
    assert df.loc["1001", "name"] == "Test Trattoria"
    assert df.loc["1001", "cuisine"] == "Italian"
    assert df.loc["1001", "source"] == "synthetic"
    assert df.loc["1001", "imported_at"] == NOW.isoformat()


def test_zero_or_blank_coordinates_become_missing(conn, nyc_csv):
    ds.import_directory_csv(conn, nyc_csv, source="synthetic")
    df = ds.load_directory(conn).set_index("nyc_id")
    assert df[["lat", "lon"]].loc["1002"].isna().all()
    assert df[["lat", "lon"]].loc["1003"].isna().all()
    assert df.loc["1003", "address"] is None


def test_directory_csv_missing_required_columns(conn, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("name,address\nA,B\n")
    with pytest.raises(ValueError, match="nyc_id"):
        ds.import_directory_csv(conn, bad, source="synthetic")


def test_mappings_and_aliases_keep_provenance(conn, nyc_csv):
    ds.import_directory_csv(conn, nyc_csv, source="synthetic")
    ds.add_mapping(conn, "1001", "yelp-fake-1", method="name_address", source="synthetic", matched_at=NOW)
    ds.add_alias(conn, "trattoria test", "1001", source="manual", created_at=NOW)
    m = ds.load_mappings(conn).iloc[0]
    assert (m.yelp_id, m.method, m.matched_at) == ("yelp-fake-1", "name_address", NOW.isoformat())
    a = ds.load_aliases(conn).iloc[0]
    assert (a.alias, a.nyc_id, a.source) == ("trattoria test", "1001", "manual")


def test_mapping_requires_known_restaurant(conn):
    with pytest.raises(Exception):
        ds.add_mapping(conn, "missing", "yelp-fake-9", method="x", source="synthetic")


def test_csv_round_trip_for_permanent_tables(conn, nyc_csv, tmp_path):
    ds.import_directory_csv(conn, nyc_csv, source="synthetic")
    ds.add_mapping(conn, "1001", "yelp-fake-1", method="name_address", source="synthetic")
    for table in ("directory", "mappings"):
        ds.export_table_csv(conn, table, tmp_path / f"{table}.csv")

    fresh = ds.connect(":memory:")
    ds.import_table_csv(fresh, "directory", tmp_path / "directory.csv")
    ds.import_table_csv(fresh, "mappings", tmp_path / "mappings.csv")
    assert len(ds.load_directory(fresh)) == 3
    assert ds.load_mappings(fresh).iloc[0].yelp_id == "yelp-fake-1"


def test_yelp_cache_cannot_be_exported_or_imported(conn, tmp_path):
    with pytest.raises(ValueError, match="24-hour cache"):
        ds.export_table_csv(conn, "yelp_cache", tmp_path / "cache.csv")
    with pytest.raises(ValueError):
        ds.import_table_csv(conn, "yelp_cache", tmp_path / "cache.csv")


def _record(yid, fetched_at, **extra):
    return {"yelp_id": yid, "name": f"Fake {yid}", "rating": 4.5, "review_count": 100,
            "price": "$$", "categories": "Italian, Pizza", "fetched_at": fetched_at, **extra}


def test_cache_expiry_boundary(conn):
    ds.upsert_cache(conn, [
        _record("fresh", NOW - timedelta(hours=23, minutes=59)),
        _record("exactly-24h", NOW - timedelta(hours=24)),
        _record("old", NOW - timedelta(days=3)),
    ], source="mock")
    fresh = ds.get_fresh_cache(conn, now=NOW)
    assert list(fresh["yelp_id"]) == ["fresh"]


def test_expired_rows_never_returned_even_before_purge(conn):
    ds.upsert_cache(conn, [_record("old", NOW - timedelta(days=2))], source="mock")
    assert ds.get_fresh_cache(conn, ["old"], now=NOW).empty
    assert ds.purge_expired(conn, now=NOW) == 1
    assert conn.execute("SELECT COUNT(*) FROM yelp_cache").fetchone()[0] == 0


def test_purge_keeps_fresh_rows(conn):
    ds.upsert_cache(conn, [_record("a", NOW), _record("b", NOW - timedelta(hours=30))], source="mock")
    assert ds.purge_expired(conn, now=NOW) == 1
    assert list(ds.get_fresh_cache(conn, now=NOW)["yelp_id"]) == ["a"]


def test_unreadable_or_naive_timestamp_counts_as_expired():
    assert ds.is_expired("not a date", NOW)
    assert ds.is_expired("2026-01-15T11:00:00", NOW)  # no timezone
    assert ds.is_expired(None, NOW)


def test_naive_datetime_rejected_on_write(conn):
    with pytest.raises(ValueError, match="timezone"):
        ds.upsert_cache(conn, [_record("x", datetime(2026, 1, 15))], source="mock")


def test_price_level_unknown_is_not_cheap(conn):
    assert ds.price_level("$$$") == 3
    assert ds.price_level(None) is None
    assert ds.price_level("") is None
    ds.upsert_cache(conn, [_record("np", NOW, price=None)], source="mock")
    row = ds.get_fresh_cache(conn, ["np"], now=NOW).iloc[0]
    assert pd.isna(row["price_level"])


def test_cache_is_stale_for_sample(conn):
    assert ds.cache_is_stale(conn, now=NOW) is False  # empty sample: nothing to refresh
    ds.add_to_sample(conn, ["a", "b"], search_area="Test Area")
    assert ds.cache_is_stale(conn, now=NOW) is True  # empty cache
    ds.upsert_cache(conn, [_record("a", NOW), _record("b", NOW - timedelta(hours=25))], source="mock")
    assert ds.cache_is_stale(conn, now=NOW) is True  # b expired
    ds.upsert_cache(conn, [_record("b", NOW)], source="mock")
    assert ds.cache_is_stale(conn, now=NOW) is False


def test_empty_cache_is_usable(conn):
    df = ds.get_fresh_cache(conn, ["anything"], now=NOW)
    assert df.empty and "price_level" in df.columns
    assert ds.get_fresh_cache(conn, [], now=NOW).empty


def test_file_database_is_created(tmp_path):
    path = tmp_path / "sub" / "db.sqlite"
    ds.connect(path).close()
    assert path.exists()
