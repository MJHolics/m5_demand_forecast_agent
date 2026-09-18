"""M5 공식 평가지표(WRMSSE) 구현. 순수 함수, 공식 정의를 그대로 따른다.

RMSSE(시리즈 하나): 예측 구간(h일) MSE를, 학습 구간의 "하루 전 값으로 예측했을 때" MSE(naive
1-step 오차의 평균제곱)로 나눠 제곱근을 취한 값. 시리즈마다 스케일(평균 판매량)이 달라도
비교 가능하게 만드는 것이 핵심.

WRMSSE: 12개 집계 레벨(전체/주/매장/카테고리/부서/품목 및 조합)마다 RMSSE를 구하고, 레벨 내에서는
최근 28일 매출액(가격 x 판매량) 비중으로 가중 평균한다. 12개 레벨은 균등 가중(1/12).

이번 단계에서는 레벨 하나(단일 시리즈 집합)에 대한 RMSSE·가중평균까지만 구현한다 — 12개 레벨
전체 계층 구성은 다음 단계(개발 순서 3번)에서 데이터 크기에 맞춰 별도로 만든다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hierarchy import LEVELS, aggregate_level, aggregate_wide_predictions, id_attrs_table
from .weights import add_dollar_sales, compute_level_weights


def rmsse(train_actuals: np.ndarray, test_actuals: np.ndarray, test_preds: np.ndarray) -> float:
    """시리즈 하나의 RMSSE. train_actuals는 학습 구간 전체(첫 판매일부터), test_actuals/test_preds는
    평가 구간(h일)."""
    train_actuals = np.asarray(train_actuals, dtype=float)
    test_actuals = np.asarray(test_actuals, dtype=float)
    test_preds = np.asarray(test_preds, dtype=float)

    # 공식 정의: 스케일(분모)은 "그 시리즈가 처음 팔린 날(첫 non-zero)"부터만 쓴다 — 출시 전
    # 리테일 진열 이전 구간의 0을 그대로 포함하면 diff가 인위적으로 0이 되어 분모가 작아지고
    # RMSSE가 과대평가된다(2026-09-17, 자체계산 0.9128 vs 공식 Kaggle 0.86967 불일치 원인 조사).
    nonzero_idx = np.flatnonzero(train_actuals)
    if len(nonzero_idx) > 0:
        train_actuals = train_actuals[nonzero_idx[0]:]

    naive_diffs = np.diff(train_actuals)
    denom = np.mean(naive_diffs ** 2)
    if denom == 0:
        # 학습 구간 내내 판매량이 전혀 안 변한 시리즈(전부 0 등) — 0으로 나누면 정의 불가.
        # 공식 대회는 이런 경우를 별도 처리하는데, 여기서는 분모가 0이면 분자도 0인지로 판정한다.
        numer = np.mean((test_actuals - test_preds) ** 2)
        return 0.0 if numer == 0 else np.inf

    h = len(test_actuals)
    numer = np.sum((test_actuals - test_preds) ** 2) / h
    return float(np.sqrt(numer / denom))


def weighted_rmsse(rmsse_by_series: dict[str, float], weight_by_series: dict[str, float]) -> float:
    """레벨 하나 안에서 시리즈별 RMSSE를 가중 평균. 가중치는 보통 최근 28일 매출액 비중(합=1)."""
    total_weight = sum(weight_by_series.values())
    if total_weight == 0:
        return float(np.mean(list(rmsse_by_series.values())))
    return sum(rmsse_by_series[k] * weight_by_series[k] for k in rmsse_by_series) / total_weight


def full_wrmsse(long_df: pd.DataFrame, pred_wide_by_id: pd.DataFrame,
                 train_end: pd.Timestamp, horizon: int = 28) -> dict:
    """12개 레벨 전부에 대해 RMSSE를 구해 균등가중(1/12) 평균한 공식 WRMSSE.

    pred_wide_by_id: index=id(item_store 최하위 키), columns=예측 대상 날짜(horizon일), 값=예측 판매량.
    상위 레벨 예측은 하위 예측의 합으로 구한다(aggregate_wide_predictions) — 실제값도 같은 규칙으로
    집계해야 비교가 성립한다.
    """
    if "dollar_sales" not in long_df.columns:
        long_df = add_dollar_sales(long_df)

    id_attrs = id_attrs_table(long_df)
    weight_start = train_end - pd.Timedelta(days=27)
    weight_end = train_end

    by_level: dict[str, float] = {}
    for name, group_cols in LEVELS:
        actual_wide = aggregate_level(long_df, group_cols, "sales")
        train_wide = actual_wide.loc[:, actual_wide.columns <= train_end]
        test_wide = actual_wide.loc[:, actual_wide.columns > train_end]

        pred_level = aggregate_wide_predictions(pred_wide_by_id, id_attrs, group_cols)
        pred_level = pred_level.reindex(index=test_wide.index, columns=test_wide.columns, fill_value=0.0)

        level_weights = compute_level_weights(long_df, group_cols, weight_start, weight_end)
        level_weights = level_weights.reindex(train_wide.index, fill_value=0.0)

        rmsse_by_series = {
            s: rmsse(train_wide.loc[s].values, test_wide.loc[s].values, pred_level.loc[s].values)
            for s in train_wide.index
        }
        by_level[name] = weighted_rmsse(rmsse_by_series, level_weights.to_dict())

    overall = float(np.mean(list(by_level.values())))
    return {"overall": overall, "by_level": by_level}
