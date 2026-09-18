import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m5fc.evaluate import full_wrmsse
from m5fc.models import seasonal_naive_forecast


def _toy_long():
    """2개 id(=item_store), 21일치. season=7 seasonal-naive가 완벽히 맞도록 7일 주기로 반복되는
    합성 패턴을 쓴다 — 베이스라인이 정말 완벽하면 WRMSSE가 0에 가까워야 한다는 걸 검증."""
    dates = pd.date_range("2024-01-01", periods=21, freq="D")
    rows = []
    for id_, store, item, state, cat, dept, price, base in [
        ("A_evaluation", "S1", "A", "CA", "FOODS", "FOODS_1", 2.0, 10),
        ("B_evaluation", "S2", "B", "TX", "HOBBIES", "HOBBIES_1", 1.0, 5),
    ]:
        for i, d in enumerate(dates):
            sales = base + (i % 7)  # 7일 주기 패턴
            rows.append({
                "id": id_, "item_id": item, "store_id": store, "cat_id": cat,
                "dept_id": dept, "state_id": state, "date": d,
                "sales": sales, "sell_price": price,
            })
    return pd.DataFrame(rows)


def test_full_wrmsse_runs_end_to_end_and_perfect_seasonal_pattern_scores_near_zero():
    long_df = _toy_long()
    train_end = pd.Timestamp("2024-01-14")  # 14일 학습, 7일 예측(패턴 주기와 일치)
    horizon = 7

    forecast = seasonal_naive_forecast(long_df, train_end, horizon=horizon, season=7)
    result = full_wrmsse(long_df, forecast, train_end, horizon=horizon)

    assert "overall" in result
    assert len(result["by_level"]) == 12
    # 완벽한 7일 주기 패턴 + 7일 seasonal-naive는 예측이 실제와 정확히 일치해야 함
    assert result["overall"] < 1e-9


def test_full_wrmsse_penalizes_bad_forecast_more_than_perfect_one():
    long_df = _toy_long()
    train_end = pd.Timestamp("2024-01-14")
    horizon = 7

    good_forecast = seasonal_naive_forecast(long_df, train_end, horizon=horizon, season=7)
    bad_forecast = good_forecast * 0  # 전부 0으로 예측하는 나쁜 베이스라인

    good_result = full_wrmsse(long_df, good_forecast, train_end, horizon=horizon)
    bad_result = full_wrmsse(long_df, bad_forecast, train_end, horizon=horizon)

    assert bad_result["overall"] > good_result["overall"]
