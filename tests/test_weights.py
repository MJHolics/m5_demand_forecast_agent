import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m5fc.weights import add_dollar_sales, compute_level_weights


def _toy_long_with_price():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
    rows = []
    # A: sales 10/day, price 2 -> dollar 20/day -> 3일 합계 60
    # B: sales 10/day, price 1 -> dollar 10/day -> 3일 합계 30
    for sid, price, sales in [("A", 2.0, 10), ("B", 1.0, 10)]:
        for d in dates:
            rows.append({"state_id": sid, "date": d, "sales": sales, "sell_price": price})
    return pd.DataFrame(rows)


def test_add_dollar_sales_multiplies_correctly():
    df = _toy_long_with_price()
    out = add_dollar_sales(df)
    assert (out[out.state_id == "A"]["dollar_sales"] == 20.0).all()
    assert (out[out.state_id == "B"]["dollar_sales"] == 10.0).all()


def test_weights_sum_to_one_and_reflect_dollar_proportion():
    df = add_dollar_sales(_toy_long_with_price())
    start, end = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-03")
    w = compute_level_weights(df, ["state_id"], start, end)
    assert abs(w.sum() - 1.0) < 1e-9
    # A의 달러 매출(60)이 B(30)의 2배이므로 가중치도 2배여야 한다
    assert abs(w["A"] - 2 * w["B"]) < 1e-9
    assert abs(w["A"] - 2 / 3) < 1e-9


def test_missing_price_treated_as_zero_dollar_sales():
    df = _toy_long_with_price()
    df.loc[df.state_id == "B", "sell_price"] = None
    out = add_dollar_sales(df)
    assert (out[out.state_id == "B"]["dollar_sales"] == 0.0).all()
