"""예측 모델. 최하위 레벨(item_store, id 기준)에서 예측하고, 상위 레벨은 합산으로 얻는다
(WRMSSE 채점 시 상위 레벨 예측은 하위 레벨 예측의 합이어야 앞뒤가 맞다)."""
from __future__ import annotations

import pandas as pd


def seasonal_naive_forecast(long_df: pd.DataFrame, train_end: pd.Timestamp, horizon: int = 28,
                             group_col: str = "id", date_col: str = "date",
                             value_col: str = "sales", season: int = 7) -> pd.DataFrame:
    """t일 예측 = t-season일 실제값. horizon이 season의 배수면 마지막 한 주기를 그대로 반복 타일링.
    가장 단순한 베이스라인 — 이것보다 못하면 모델을 쓸 이유가 없다는 기준선."""
    train = long_df[long_df[date_col] <= train_end]
    last_season = train[train[date_col] > train_end - pd.Timedelta(days=season)]
    last_season_wide = last_season.pivot(index=group_col, columns=date_col, values=value_col)
    last_season_wide = last_season_wide.sort_index(axis=1)

    forecast_dates = pd.date_range(train_end + pd.Timedelta(days=1), periods=horizon)
    n_tiles = -(-horizon // season)  # ceil
    tiled_cols = list(last_season_wide.columns) * n_tiles
    tiled = pd.concat([last_season_wide] * n_tiles, axis=1)
    tiled.columns = tiled_cols
    tiled = tiled.iloc[:, :horizon]
    tiled.columns = forecast_dates
    return tiled
