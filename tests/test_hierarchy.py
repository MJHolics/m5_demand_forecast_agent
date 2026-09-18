import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m5fc.hierarchy import aggregate_level, build_all_levels, LEVELS, EXPECTED_SERIES_COUNT


def _toy_long():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02"])
    rows = []
    for state, store, cat, item, sales_vals in [
        ("CA", "CA_1", "FOODS", "FOODS_1", [1, 2]),
        ("CA", "CA_1", "HOBBIES", "HOBBIES_1", [3, 4]),
        ("TX", "TX_1", "FOODS", "FOODS_1", [5, 6]),
    ]:
        for d, v in zip(dates, sales_vals):
            rows.append({
                "state_id": state, "store_id": store, "cat_id": cat,
                "dept_id": cat, "item_id": item, "date": d, "sales": v,
            })
    return pd.DataFrame(rows)


def test_total_level_sums_everything():
    long_df = _toy_long()
    out = aggregate_level(long_df, [], "sales")
    d1 = pd.Timestamp("2024-01-01")
    d2 = pd.Timestamp("2024-01-02")
    # day1: 1+3+5=9, day2: 2+4+6=12
    assert out.loc["total", d1] == 9
    assert out.loc["total", d2] == 12


def test_state_level_groups_correctly():
    long_df = _toy_long()
    out = aggregate_level(long_df, ["state_id"], "sales")
    d1 = pd.Timestamp("2024-01-01")
    # CA day1 = FOODS(1) + HOBBIES(3) = 4; TX day1 = 5
    assert out.loc["CA", d1] == 4
    assert out.loc["TX", d1] == 5


def test_item_store_level_matches_bottom_granularity():
    long_df = _toy_long()
    out = aggregate_level(long_df, ["item_id", "store_id"], "sales")
    d1 = pd.Timestamp("2024-01-01")
    assert out.loc["FOODS_1_CA_1", d1] == 1
    assert out.loc["HOBBIES_1_CA_1", d1] == 3
    assert out.loc["FOODS_1_TX_1", d1] == 5


def test_build_all_levels_returns_12_levels():
    long_df = _toy_long()
    result = build_all_levels(long_df)
    assert len(result) == 12
    assert set(result.keys()) == {name for name, _ in LEVELS}


def test_expected_series_count_matches_official_m5_spec():
    total = sum(EXPECTED_SERIES_COUNT.values())
    assert total == 42840  # M5 공식 문서에 명시된 전체 시리즈 수
