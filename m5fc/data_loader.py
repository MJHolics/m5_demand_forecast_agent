"""M5 원본 CSV -> 표 형태(tabular long) 변환.

원본은 wide 포맷(시리즈 1행 x 날짜 1,941열)이라 그대로는 lag/rolling 특징을 못 만든다.
멜트(long) 후 메모리를 다운캐스팅해 30,490시리즈 x ~1,941일(약 5,900만 행)을 다룰 수 있게 한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_calendar(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    cal = pd.read_csv(data_dir / "calendar.csv", parse_dates=["date"])
    for c in ("event_name_1", "event_type_1", "event_name_2", "event_type_2"):
        cal[c] = cal[c].fillna("none").astype("category")
    for c in ("wday", "month", "year", "snap_CA", "snap_TX", "snap_WI"):
        cal[c] = cal[c].astype("int16")
    return cal


def load_prices(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    prices = pd.read_csv(data_dir / "sell_prices.csv")
    prices["store_id"] = prices["store_id"].astype("category")
    prices["item_id"] = prices["item_id"].astype("category")
    prices["wm_yr_wk"] = prices["wm_yr_wk"].astype("int32")
    prices["sell_price"] = prices["sell_price"].astype("float32")
    return prices


def melt_sales(sales_wide: pd.DataFrame, id_vars: list[str] = ID_COLS) -> pd.DataFrame:
    """wide(시리즈 1행) -> long(시리즈-일 1행). d_1, d_2, ... 컬럼만 melt 대상."""
    d_cols = [c for c in sales_wide.columns if c.startswith("d_")]
    long = sales_wide.melt(id_vars=id_vars, value_vars=d_cols, var_name="d", value_name="sales")
    long["sales"] = long["sales"].astype("int16")
    for c in id_vars:
        long[c] = long[c].astype("category")
    return long


def build_long_table(data_dir: str | Path, sales_csv: str = "sales_train_evaluation.csv") -> pd.DataFrame:
    """long(시리즈-일) + calendar + price를 합친 단일 표. 리키지 없음 — 병합만 하고 특징은 features.py 담당."""
    data_dir = Path(data_dir)
    sales_wide = pd.read_csv(data_dir / sales_csv)
    long = melt_sales(sales_wide)

    cal = load_calendar(data_dir)
    long = long.merge(
        cal[["d", "date", "wm_yr_wk", "wday", "month", "year",
             "event_name_1", "event_type_1", "event_name_2", "event_type_2",
             "snap_CA", "snap_TX", "snap_WI"]],
        on="d", how="left",
    )

    prices = load_prices(data_dir)
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    long["snap"] = np.select(
        [long["state_id"] == "CA", long["state_id"] == "TX", long["state_id"] == "WI"],
        [long["snap_CA"], long["snap_TX"], long["snap_WI"]],
        default=0,
    ).astype("int16")
    long = long.drop(columns=["snap_CA", "snap_TX", "snap_WI"])

    # d/wm_yr_wk는 병합용 키였을 뿐 이후 단계에서 안 쓴다 — d는 문자열(object dtype)이라
    # 5,900만 행 규모에서 메모리 부담이 특히 크다.
    long = long.drop(columns=["d", "wm_yr_wk"])
    return long.sort_values(["id", "date"]).reset_index(drop=True)
