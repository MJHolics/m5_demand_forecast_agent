import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m5fc.features import (
    add_lag_features, add_rolling_features, add_price_features,
    build_feature_frame, usable_feature_columns,
)


def _toy_df():
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "id": ["A"] * 10,
        "date": dates,
        "sales": np.arange(10, dtype=float),  # 0,1,2,...,9
    })


def test_lag_1_shifts_by_one_no_lookahead():
    df = _toy_df()
    out = add_lag_features(df, lags=(1,))
    assert out.loc[5, "lag_1"] == 4
    assert out.loc[0, "lag_1"] != out.loc[0, "sales"]
    assert pd.isna(out.loc[0, "lag_1"])


def test_lag_does_not_mix_across_series():
    df = pd.concat([_toy_df().assign(id="A"), _toy_df().assign(id="B", sales=lambda d: d.sales + 100)])
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    out = add_lag_features(df, lags=(1,))
    b_first_row = out[out.id == "B"].iloc[0]
    assert pd.isna(b_first_row["lag_1"])  # B의 첫 행이 A의 마지막 값을 lag로 가져오면 리키지


def test_rolling_mean_excludes_current_value():
    df = _toy_df()
    out = add_rolling_features(df, windows=(3,))
    # t=5(sales=5) 기준 roll_mean_3은 sales[2],sales[3],sales[4] = (2+3+4)/3 = 3.0 (자기 자신 5 미포함)
    assert abs(out.loc[5, "roll_mean_3"] - 3.0) < 1e-9


def test_price_change_pct_uses_previous_day_price_not_today():
    df = _toy_df()
    df["sell_price"] = [10.0, 10.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0]
    out = add_price_features(df)
    # t=2에서 가격이 10->12로 뛰었다. price_change_pct[2]는 (12-10)/10=0.2여야 한다(전일 대비).
    assert abs(out.loc[2, "price_change_pct"] - 0.2) < 1e-9
    # t=3(가격 변화 없음)의 price_change_pct는 0이어야 한다 — t=2의 값이 새어 들어오면 리키지.
    assert abs(out.loc[3, "price_change_pct"] - 0.0) < 1e-9


def test_build_feature_frame_runs_end_to_end():
    df = _toy_df()
    out = build_feature_frame(df)
    assert "days_since_start" in out.columns
    assert out["days_since_start"].iloc[0] == 0
    assert out["days_since_start"].iloc[-1] == 9
    assert "dow" in out.columns


def test_extra_shift_pushes_lag_further_back_for_direct_multihorizon():
    """지평선별 직접 예측: extra_shift=h-1이면 lag_1이 실제로는 h칸 전을 봐야 한다 —
    t=8(sales=8) 기준 extra_shift=6(h=7)일 때 lag_1은 t=1(sales=1)이어야 한다(shift 1+6=7)."""
    df = _toy_df()
    out = add_lag_features(df, lags=(1,), extra_shift=6)
    assert out.loc[8, "lag_1"] == 1  # 8-7=1
    # extra_shift=0(기존 동작)이면 t=8의 lag_1은 t=7(sales=7)
    out0 = add_lag_features(df, lags=(1,), extra_shift=0)
    assert out0.loc[8, "lag_1"] == 7


def test_extra_shift_column_names_stay_the_same_across_horizons():
    """모델 재사용을 위해 extra_shift가 달라도 컬럼 이름(lag_1 등)은 그대로여야 한다 —
    같은 feature_cols 리스트로 다른 지평선 모델에 X를 넣을 수 있어야 하기 때문."""
    df = _toy_df()
    out_h1 = build_feature_frame(df, lags=(1, 7), windows=(3,), extra_shift=0)
    out_h7 = build_feature_frame(df, lags=(1, 7), windows=(3,), extra_shift=6)
    assert set(usable_feature_columns(out_h1)) == set(usable_feature_columns(out_h7))


def test_usable_feature_columns_picks_up_custom_lags():
    """demand_forecasting_gbm에서 커스텀 lag가 하드코딩 목록 때문에 조용히 빠졌던 사고를
    처음부터 재현 방지 — 이 프로젝트에서도 동일한 회귀 테스트를 먼저 박아둔다."""
    df = _toy_df()
    out = build_feature_frame(df, lags=(1, 2, 3, 12), windows=(3, 12))
    cols = usable_feature_columns(out)
    assert "lag_12" in cols
    assert "lag_2" in cols
    assert "roll_mean_12" in cols
    assert "lag_7" not in cols  # 애초에 안 만들어졌으니 없어야 함
