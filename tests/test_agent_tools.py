import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m5fc.agent_tools import find_series, get_forecast, get_recent_trend, get_context, MAX_MATCHED_SERIES


def _toy_artifact():
    return {
        "model": "seasonal_naive_7day",
        "train_end": "2016-04-24",
        "forecast_dates": ["2016-04-25", "2016-04-26"],
        "events_in_forecast_window": [{"date": "2016-04-25", "event_name_1": "Easter", "event_type_1": "Religious"}],
        "snap_days_by_state": {"CA": ["2016-04-25"], "TX": [], "WI": []},
        "series": {
            "A": {"item_id": "FOODS_1_001", "store_id": "CA_1", "cat_id": "FOODS", "dept_id": "FOODS_1",
                  "state_id": "CA", "forecast": {"2016-04-25": 3.0, "2016-04-26": 4.0},
                  "recent_actual": {"2016-04-23": 2.0, "2016-04-24": 3.0}, "current_price": 2.0},
            "B": {"item_id": "FOODS_1_002", "store_id": "CA_1", "cat_id": "FOODS", "dept_id": "FOODS_1",
                  "state_id": "CA", "forecast": {"2016-04-25": 1.0, "2016-04-26": 1.5},
                  "recent_actual": {"2016-04-23": 1.0, "2016-04-24": 1.0}, "current_price": 3.0},
            "C": {"item_id": "HOBBIES_1_001", "store_id": "TX_1", "cat_id": "HOBBIES", "dept_id": "HOBBIES_1",
                  "state_id": "TX", "forecast": {"2016-04-25": 5.0, "2016-04-26": 5.0},
                  "recent_actual": {"2016-04-23": 5.0, "2016-04-24": 5.0}, "current_price": 10.0},
        },
    }


def test_find_series_filters_by_state_and_category():
    art = _toy_artifact()
    matched = find_series(art, state_id="CA", cat_id="FOODS")
    assert set(matched) == {"A", "B"}


def test_find_series_with_no_filters_returns_empty():
    art = _toy_artifact()
    assert find_series(art) == []


def test_get_forecast_sums_across_matched_series():
    art = _toy_artifact()
    totals = get_forecast(art, ["A", "B"])
    assert totals["2016-04-25"] == 4.0  # 3.0 + 1.0
    assert totals["2016-04-26"] == 5.5  # 4.0 + 1.5


def test_get_recent_trend_sums_correctly():
    art = _toy_artifact()
    trend = get_recent_trend(art, ["A", "B"])
    assert trend["2016-04-23"] == 3.0
    assert trend["2016-04-24"] == 4.0


def test_get_context_averages_price_and_scopes_snap_to_relevant_states():
    art = _toy_artifact()
    ctx = get_context(art, ["A", "B"])
    assert ctx["avg_current_price"] == 2.5
    assert ctx["series_count"] == 2
    assert ctx["snap_days_by_state"] == {"CA": ["2016-04-25"]}  # TX 시리즈가 안 섞여 있으니 CA만


def test_get_context_includes_multiple_states_when_series_span_them():
    art = _toy_artifact()
    ctx = get_context(art, ["A", "C"])  # CA + TX
    assert set(ctx["snap_days_by_state"].keys()) == {"CA", "TX"}


def test_max_matched_series_does_not_silently_truncate_realistic_category_queries():
    """회귀 테스트 — 예전 상한(2000)에서는 "CA+FOODS"처럼 실제로 2000개 넘게 매칭되는 흔한
    질의의 합계가 조용히 잘렸다(전체 시리즈 30,490개 중 상한이 그보다 훨씬 작았음). 상한은
    최소한 전체 item_store 시리즈 수(30,490)보다 커야, 필터가 넓어도 안 잘린다."""
    assert MAX_MATCHED_SERIES >= 30490
