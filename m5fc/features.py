"""M5 특징 엔지니어링. 전부 순수 함수, 미래 정보 누출 없음.

입력 규약: DataFrame에 최소 [id, date, sales] 컬럼이 있고 (id, date) 오름차순 정렬.
모든 lag/rolling 특징은 t 시점 예측에 t-1 이하 값만 쓴다(shift(1) 이후 계산) —
demand_forecasting_gbm의 리키지 회귀 테스트 관행을 그대로 따른다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LAG_DAYS_DEFAULT = (1, 7, 14, 28)
ROLL_WINDOWS_DEFAULT = (7, 28)


def add_lag_features(df: pd.DataFrame, lags=LAG_DAYS_DEFAULT, group_col: str = "id", value_col: str = "sales",
                      extra_shift: int = 0) -> pd.DataFrame:
    """extra_shift>0: 지평선별 직접(non-recursive) 예측용 — lag_k 컬럼 이름은 그대로 두되 실제
    shift량은 k+extra_shift를 쓴다. h일 뒤를 예측하는 모델은 extra_shift=h-1을 넘겨서, 그 모델이
    "예측 시점 기준 h일 전(=학습 시점 기준으로는 항상 확보돼 있는 값)"만 보게 만든다."""
    out = df.copy()
    g = out.groupby(group_col, observed=True)[value_col]
    for k in lags:
        out[f"lag_{k}"] = g.shift(k + extra_shift)
    return out


def add_rolling_features(df: pd.DataFrame, windows=ROLL_WINDOWS_DEFAULT, group_col: str = "id", value_col: str = "sales",
                          extra_shift: int = 0) -> pd.DataFrame:
    """rolling은 shift(1+extra_shift) 뒤에 계산한다 — t 시점 값 자신이 자기 통계에 들어가면 리키지."""
    out = df.copy()
    shifted = out.groupby(group_col, observed=True)[value_col].shift(1 + extra_shift)
    for w in windows:
        out[f"roll_mean_{w}"] = shifted.groupby(out[group_col], observed=True).transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).mean()
        )
        out[f"roll_std_{w}"] = shifted.groupby(out[group_col], observed=True).transform(
            lambda s: s.rolling(w, min_periods=max(2, w // 3)).std()
        )
    return out


def add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    out["dow"] = d.dt.dayofweek.astype("int16")
    out["is_weekend"] = (out["dow"] >= 5).astype("int16")
    out["day_of_year"] = d.dt.dayofyear.astype("int16")
    return out


def add_trend_index(df: pd.DataFrame, group_col: str = "id", date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    d = pd.to_datetime(out[date_col])
    start = d.groupby(out[group_col], observed=True).transform("min")
    out["days_since_start"] = (d - start).dt.days.astype("int32")
    return out


def add_price_features(df: pd.DataFrame, group_col: str = "id", price_col: str = "sell_price") -> pd.DataFrame:
    """가격 변화 특징. lag/rolling과 동일하게 shift(1) 이후 계산 — 오늘 가격을 오늘 예측에 그대로
    쓰는 건 리키지가 아니다(가격은 판매 이전에 결정되는 외생변수)지만, 가격 '변화율'은 전일 대비로
    안전하게 계산한다."""
    out = df.copy()
    g = out.groupby(group_col, observed=True)[price_col]
    prev_price = g.shift(1)
    out["price_change_pct"] = ((out[price_col] - prev_price) / prev_price).replace([np.inf, -np.inf], 0).fillna(0)
    roll_mean_price = g.transform(lambda s: s.shift(1).rolling(28, min_periods=1).mean())
    out["price_vs_roll28"] = (out[price_col] / roll_mean_price).replace([np.inf, -np.inf], 1).fillna(1)
    return out


FEATURE_COLUMNS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_std_7", "roll_mean_28", "roll_std_28",
    "dow", "is_weekend", "day_of_year", "days_since_start",
    "wday", "month", "year", "snap",
    "sell_price", "price_change_pct", "price_vs_roll28",
]


def build_feature_frame(df: pd.DataFrame, lags=LAG_DAYS_DEFAULT, windows=ROLL_WINDOWS_DEFAULT,
                         group_col: str = "id", date_col: str = "date", value_col: str = "sales",
                         extra_shift: int = 0) -> pd.DataFrame:
    out = df.sort_values([group_col, date_col]).reset_index(drop=True)
    out = add_lag_features(out, lags, group_col, value_col, extra_shift)
    out = add_rolling_features(out, windows, group_col, value_col, extra_shift)
    out = add_calendar_features(out, date_col)
    out = add_trend_index(out, group_col, date_col)
    if "sell_price" in out.columns:
        out = add_price_features(out, group_col)
    return out


def usable_feature_columns(df: pd.DataFrame) -> list[str]:
    """FEATURE_COLUMNS 우선이되, build_feature_frame이 실제로 만든(커스텀 lags/windows 포함)
    특징 컬럼을 동적으로 잡는다 — demand_forecasting_gbm에서 겪은 하드코딩 버그를 처음부터 피한다."""
    dynamic = [c for c in df.columns if c.startswith("lag_") or c.startswith("roll_mean_") or c.startswith("roll_std_")]
    static = [c for c in FEATURE_COLUMNS if not (c.startswith("lag_") or c.startswith("roll_"))]
    ordered = dynamic + static
    seen = set()
    result = []
    for c in ordered:
        if c in df.columns and c not in seen:
            seen.add(c)
            result.append(c)
    return result
