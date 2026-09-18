"""LightGBM 학습 + 재귀적(recursive) 다단계 예측.

재귀 방식을 택한 이유: lag_1처럼 "어제 값"에 의존하는 특징은 예측 1일차 이후엔 실제값이 없다.
직접 28개 지평선별 모델(non-recursive multi-horizon)이 오차 누적을 막는 데는 더 낫다고 알려져
있지만, 이번 단계에서는 구현이 단순한 재귀 방식으로 먼저 파이프라인을 완성하고, 오차 누적 정도를
직접 재서(정직한 범위) 다음 단계에서 개선 여부를 판단한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .features import build_feature_frame, usable_feature_columns

CATEGORY_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]


def prepare_training_frame(long_df: pd.DataFrame, train_end: pd.Timestamp,
                            lags=(1, 7, 14, 28), windows=(7, 28)) -> tuple[pd.DataFrame, list[str]]:
    """학습 구간(date <= train_end)에 대해서만 특징을 만든다. lag/rolling은 train_end 이전 값만
    쓰므로 그 자체로 리키지가 없지만, 행 자체를 train_end 이후로 넓히지 않는 것이 이중 안전장치."""
    hist = long_df[long_df["date"] <= train_end]
    feat = build_feature_frame(hist, lags=lags, windows=windows)
    feature_cols = usable_feature_columns(feat)
    return feat, feature_cols


def train_lightgbm(feat_df: pd.DataFrame, feature_cols: list[str], target_col: str = "sales",
                    **lgbm_kwargs) -> LGBMRegressor:
    """행 선택(dropna)과 열 선택(feature_cols)을 한 번에 슬라이스해 전체 폭 복사본을 따로
    만들지 않는다 — 5,900만 행 규모에서 복사본 하나가 통째로 GB 단위라 메모리 문제(OOM)로
    이어졌던 걸 고치며 알게 됐다(에러 대처 기록)."""
    mask = feat_df["lag_1"].notna()  # 초기 lag 확보 안 된 시리즈 시작 구간 제외
    X = feat_df.loc[mask, feature_cols].copy()
    for c in CATEGORY_COLS:
        if c in X.columns:
            X[c] = X[c].cat.codes
    y = feat_df.loc[mask, target_col]
    params = dict(n_estimators=200, learning_rate=0.05, num_leaves=63, random_state=42, verbosity=-1)
    params.update(lgbm_kwargs)
    model = LGBMRegressor(**params)
    model.fit(X, y)
    return model


def train_direct_multihorizon(long_df: pd.DataFrame, train_end: pd.Timestamp, horizon: int = 28,
                               lags=(1, 7, 14, 28), windows=(7, 28),
                               min_train_date: pd.Timestamp | None = None,
                               **lgbm_kwargs) -> dict[int, tuple[LGBMRegressor, list[str]]]:
    """지평선(h=1..horizon)마다 별도 모델. 재귀 없음 — 각 모델은 extra_shift=h-1로 "h일 전까지만
    보이는" 특징을 학습해서, 예측 시점(train_end+h)의 특징이 전부 train_end 이하 실제값으로만
    채워지게 만든다. 오차가 다음 날로 전파되지 않는다.

    min_train_date: 학습 구간을 최근 N일로 제한(속도). None이면 전체 이력을 쓴다.

    메모리 참고(에러 대처 기록, 2026-09-16): 처음엔 전체 이력(5,900만 행)으로 28개 모델을
    돌리다 OOM으로 죽었다. 원인은 두 가지였다 — ① `long_df`에 특징공학에 안 쓰는 넓은 컬럼
    (item_id 등 id 계열, 문자열 `d`)까지 실려 있어 매 반복 특징 프레임이 불필요하게 컸고,
    ② 반복 사이에 이전 `feat`를 명시적으로 안 비워 GC가 늦게 돌 여지가 있었다. 아래에서
    ①은 호출 전 `hist`를 필요한 컬럼만으로 슬림화, ②는 매 반복 `del`+`gc.collect()`로 고쳤다."""
    import gc

    hist = long_df[long_df["date"] <= train_end]
    if min_train_date is not None:
        hist = hist[hist["date"] >= min_train_date]
    hist = hist[["id", "date", "sales", "sell_price",
                 "event_name_1", "event_type_1", "event_name_2", "event_type_2",
                 "wday", "month", "year", "snap"]].copy()

    models: dict[int, tuple[LGBMRegressor, list[str]]] = {}
    for h in range(1, horizon + 1):
        feat = build_feature_frame(hist, lags=lags, windows=windows, extra_shift=h - 1)
        feature_cols = usable_feature_columns(feat)
        model = train_lightgbm(feat, feature_cols, **lgbm_kwargs)
        models[h] = (model, feature_cols)
        del feat
        gc.collect()
    return models


def direct_multihorizon_forecast(models: dict[int, tuple[LGBMRegressor, list[str]]], long_df: pd.DataFrame,
                                  train_end: pd.Timestamp, horizon: int = 28,
                                  lags=(1, 7, 14, 28), windows=(7, 28),
                                  group_col: str = "id", date_col: str = "date",
                                  value_col: str = "sales") -> pd.DataFrame:
    """각 지평선 모델에 "train_end까지의 실제 데이터 + 예측하려는 그 하루(값은 비워둠)"만 주고
    예측한다. 서로 다른 h끼리 의존관계가 없어 병렬화 가능하지만, 여기서는 단순 순차 처리."""
    lookback_days = max(max(lags), max(windows)) + horizon + 7
    hist = long_df[
        (long_df[date_col] <= train_end) & (long_df[date_col] > train_end - pd.Timedelta(days=lookback_days))
    ].copy()

    static_cols = [c for c in hist.columns if c not in (date_col, value_col, group_col)]
    id_to_static = hist.drop_duplicates(group_col).set_index(group_col)[static_cols]

    forecast_dates = pd.date_range(train_end + pd.Timedelta(days=1), periods=horizon)
    ids = long_df[group_col].unique()
    preds = pd.DataFrame(index=ids, columns=forecast_dates, dtype=float)

    for h, day in enumerate(forecast_dates, start=1):
        model, feature_cols = models[h]
        new_row = id_to_static.copy()
        new_row[date_col] = day
        new_row[value_col] = np.nan
        new_row = new_row.reset_index().rename(columns={"index": group_col})

        step = pd.concat([hist, new_row], ignore_index=True)
        step_feat = build_feature_frame(step, lags=lags, windows=windows, extra_shift=h - 1,
                                         group_col=group_col, date_col=date_col, value_col=value_col)
        today_rows = step_feat[step_feat[date_col] == day].set_index(group_col)

        X_today = today_rows[feature_cols].copy()
        for c in CATEGORY_COLS:
            if c in X_today.columns:
                X_today[c] = X_today[c].cat.codes
        yhat = np.clip(model.predict(X_today), 0, None)
        preds.loc[today_rows.index, day] = yhat

    return preds


def recursive_forecast(model: LGBMRegressor, long_df: pd.DataFrame, feature_cols: list[str],
                        train_end: pd.Timestamp, horizon: int = 28,
                        lags=(1, 7, 14, 28), windows=(7, 28),
                        group_col: str = "id", date_col: str = "date",
                        value_col: str = "sales") -> pd.DataFrame:
    """하루씩 예측하고 그 예측값을 다음 날 lag 계산에 실제값처럼 넣어 반복(재귀).
    lookback은 max(lags, windows)*2 + 여유만큼만 들고 다녀 매 스텝 특징 재계산 비용을 줄인다."""
    lookback_days = max(max(lags), max(windows)) * 2 + 7
    working = long_df[
        (long_df[date_col] <= train_end) & (long_df[date_col] > train_end - pd.Timedelta(days=lookback_days))
    ].copy()

    forecast_dates = pd.date_range(train_end + pd.Timedelta(days=1), periods=horizon)
    ids = long_df[group_col].unique()
    preds = pd.DataFrame(index=ids, columns=forecast_dates, dtype=float)

    static_cols = [c for c in working.columns if c not in (date_col, value_col, group_col)]
    id_to_static = working.drop_duplicates(group_col).set_index(group_col)[static_cols]

    for day in forecast_dates:
        new_rows = id_to_static.copy()
        new_rows[date_col] = day
        new_rows[value_col] = np.nan
        new_rows = new_rows.reset_index().rename(columns={"index": group_col})

        step = pd.concat([working, new_rows], ignore_index=True)
        step_feat = build_feature_frame(step, lags=lags, windows=windows,
                                         group_col=group_col, date_col=date_col, value_col=value_col)
        today_rows = step_feat[step_feat[date_col] == day].set_index(group_col)

        X_today = today_rows[feature_cols].copy()
        for c in CATEGORY_COLS:
            if c in X_today.columns:
                X_today[c] = X_today[c].cat.codes
        yhat = model.predict(X_today)
        yhat = np.clip(yhat, 0, None)  # 판매량은 음수가 될 수 없다

        preds.loc[today_rows.index, day] = yhat

        new_rows = new_rows.set_index(group_col)
        new_rows.loc[today_rows.index, value_col] = yhat
        new_rows = new_rows.reset_index()
        working = pd.concat([working, new_rows], ignore_index=True)
        cutoff = day - pd.Timedelta(days=lookback_days)
        working = working[working[date_col] > cutoff]

    return preds
