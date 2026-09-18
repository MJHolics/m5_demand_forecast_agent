"""WRMSSE 가중치 — 공식 정의: 학습 구간 마지막 28일의 매출액(판매량 x 가격) 비중, 레벨 내 합=1.
(출처: M5 공식 대회 가이드 — "weights computed from the last 28 observations of the training sample,
cumulative actual dollar sales in that period". 12개 레벨은 균등 가중 1/12.)
"""
from __future__ import annotations

import pandas as pd

from .hierarchy import aggregate_level, LEVELS


def add_dollar_sales(long_df: pd.DataFrame, sales_col: str = "sales",
                      price_col: str = "sell_price") -> pd.DataFrame:
    out = long_df.copy()
    out["dollar_sales"] = (out[sales_col].astype(float) * out[price_col].astype(float)).fillna(0.0)
    return out


def compute_level_weights(long_df: pd.DataFrame, group_cols: list[str],
                           weight_start: pd.Timestamp, weight_end: pd.Timestamp,
                           date_col: str = "date") -> pd.Series:
    """레벨 하나의 시리즈별 가중치(합=1). weight_start~weight_end(포함) 구간의 달러 매출 비중."""
    window = long_df[(long_df[date_col] >= weight_start) & (long_df[date_col] <= weight_end)]
    dollar_wide = aggregate_level(window, group_cols, "dollar_sales", date_col)
    totals = dollar_wide.sum(axis=1)
    grand_total = totals.sum()
    if grand_total == 0:
        # 극단적 예외 상황(윈도우 안에 매출이 전혀 없음) — 균등 가중으로 대체
        return pd.Series(1.0 / len(totals), index=totals.index)
    return totals / grand_total


def compute_all_level_weights(long_df: pd.DataFrame, weight_start: pd.Timestamp,
                               weight_end: pd.Timestamp) -> dict[str, pd.Series]:
    return {
        name: compute_level_weights(long_df, cols, weight_start, weight_end)
        for name, cols in LEVELS
    }
