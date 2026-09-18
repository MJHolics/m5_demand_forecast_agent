import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m5fc.models_gbm import (
    prepare_training_frame, train_lightgbm, recursive_forecast,
    train_direct_multihorizon, direct_multihorizon_forecast,
)


def _toy_long(n_days=60):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for id_, store, item, state, cat, dept, price, base in [
        ("A_evaluation", "S1", "A", "CA", "FOODS", "FOODS_1", 2.0, 10),
        ("B_evaluation", "S2", "B", "TX", "HOBBIES", "HOBBIES_1", 1.0, 5),
    ]:
        for i, d in enumerate(dates):
            sales = base + (i % 7)
            rows.append({
                "id": id_, "item_id": item, "store_id": store, "cat_id": cat,
                "dept_id": dept, "state_id": state, "date": d,
                "sales": sales, "sell_price": price,
                "event_name_1": "none", "event_type_1": "none",
                "event_name_2": "none", "event_type_2": "none",
                "wday": d.dayofweek, "month": d.month, "year": d.year, "snap": 0,
            })
    return pd.DataFrame(rows)


def test_prepare_and_train_runs_end_to_end():
    long_df = _toy_long()
    train_end = pd.Timestamp("2024-02-15")
    feat, cols = prepare_training_frame(long_df, train_end, lags=(1, 7), windows=(7,))
    assert len(cols) > 0
    model = train_lightgbm(feat, cols, n_estimators=10)
    assert model is not None


def test_recursive_forecast_produces_correct_shape_and_nonnegative():
    long_df = _toy_long()
    train_end = pd.Timestamp("2024-02-15")
    lags, windows = (1, 7), (7,)
    feat, cols = prepare_training_frame(long_df, train_end, lags=lags, windows=windows)
    model = train_lightgbm(feat, cols, n_estimators=10)

    preds = recursive_forecast(model, long_df, cols, train_end, horizon=7, lags=lags, windows=windows)
    assert preds.shape == (2, 7)
    assert (preds.values >= 0).all()
    assert not preds.isna().any().any()


def test_direct_multihorizon_runs_end_to_end_and_produces_valid_predictions():
    long_df = _toy_long()
    train_end = pd.Timestamp("2024-02-15")
    lags, windows = (1, 7), (7,)

    models = train_direct_multihorizon(long_df, train_end, horizon=7, lags=lags, windows=windows, n_estimators=10)
    assert len(models) == 7

    preds = direct_multihorizon_forecast(models, long_df, train_end, horizon=7, lags=lags, windows=windows)
    assert preds.shape == (2, 7)
    assert (preds.values >= 0).all()
    assert not preds.isna().any().any()
