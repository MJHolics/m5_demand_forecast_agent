"""M5 공식 12개 집계 레벨. 순수 함수 — long 표를 받아 레벨별로 (series_key x date) 피벗한다.

12개 레벨 정의는 M5 공식 문서 그대로: Total(1) - State(3) - Store(10) - Category(3) -
Dept(7) - State*Category(9) - State*Dept(21) - Store*Category(30) - Store*Dept(70) -
Item(3049) - Item*State(9147) - Item*Store(30490). 총 42,840개 시리즈.
"""
from __future__ import annotations

import pandas as pd

LEVELS: list[tuple[str, list[str]]] = [
    ("total", []),
    ("state", ["state_id"]),
    ("store", ["store_id"]),
    ("category", ["cat_id"]),
    ("dept", ["dept_id"]),
    ("state_category", ["state_id", "cat_id"]),
    ("state_dept", ["state_id", "dept_id"]),
    ("store_category", ["store_id", "cat_id"]),
    ("store_dept", ["store_id", "dept_id"]),
    ("item", ["item_id"]),
    ("item_state", ["item_id", "state_id"]),
    ("item_store", ["item_id", "store_id"]),
]

EXPECTED_SERIES_COUNT = {
    "total": 1, "state": 3, "store": 10, "category": 3, "dept": 7,
    "state_category": 9, "state_dept": 21, "store_category": 30, "store_dept": 70,
    "item": 3049, "item_state": 9147, "item_store": 30490,
}


def _series_key(row_index) -> str:
    """groupby 결과 인덱스(튜플 또는 스칼라)를 사람이 읽을 수 있는 문자열 키로."""
    if isinstance(row_index, tuple):
        return "_".join(str(v) for v in row_index)
    return str(row_index)


def aggregate_level(long_df: pd.DataFrame, group_cols: list[str], value_col: str,
                     date_col: str = "date") -> pd.DataFrame:
    """long 표 -> (series_key x date) 피벗, value_col을 합산 집계.
    group_cols가 빈 리스트면 전체를 시리즈 하나("total")로 합산한다."""
    if not group_cols:
        agg = long_df.groupby(date_col, observed=True)[value_col].sum()
        wide = agg.to_frame().T
        wide.index = ["total"]
        return wide

    agg = long_df.groupby(group_cols + [date_col], observed=True)[value_col].sum().unstack(date_col)
    agg.index = [_series_key(ix) for ix in agg.index]
    return agg.fillna(0.0)


def build_all_levels(long_df: pd.DataFrame, value_col: str = "sales",
                      date_col: str = "date") -> dict[str, pd.DataFrame]:
    """12개 레벨 전부를 (series_key x date) 피벗 딕셔너리로."""
    return {name: aggregate_level(long_df, cols, value_col, date_col) for name, cols in LEVELS}


def id_attrs_table(long_df: pd.DataFrame, id_col: str = "id") -> pd.DataFrame:
    """id(item_store 최하위 키)별 속성(item_id·store_id·cat_id·dept_id·state_id) 매핑표.
    long_df에서 id당 값이 유일한 컬럼들만 뽑아 중복 제거."""
    attr_cols = ["item_id", "store_id", "cat_id", "dept_id", "state_id"]
    return long_df[[id_col] + attr_cols].drop_duplicates(id_col).set_index(id_col)


def aggregate_wide_predictions(pred_wide: pd.DataFrame, id_attrs: pd.DataFrame,
                                group_cols: list[str]) -> pd.DataFrame:
    """예측치 피벗표(index=id, columns=date)를 상위 레벨로 합산. 하위(item_store) 예측의 합이
    상위 레벨 예측이어야 앞뒤가 맞는다 — 실제(aggregate_level)와 동일한 series_key 규칙을 쓴다."""
    if not group_cols:
        summed = pred_wide.sum(axis=0).to_frame().T
        summed.index = ["total"]
        return summed

    merged = pred_wide.join(id_attrs, how="left")
    date_cols = list(pred_wide.columns)
    agg = merged.groupby(group_cols, observed=True)[date_cols].sum()
    agg.index = [_series_key(ix) for ix in agg.index]
    return agg
