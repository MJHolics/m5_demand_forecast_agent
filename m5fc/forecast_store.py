"""예측 결과 + 근거 데이터를 미리 계산해 JSON으로 저장 — 에이전트·제품 배포 양쪽에서 재사용.

이번 세션 결론(seasonal-naive가 LightGBM 5종을 전부 이김)에 따라 naive 예측을 "현재 최선"으로
저장한다. 나중에 더 나은 모델이 나오면 이 파일만 다시 만들면 되고, 에이전트·프론트는 안 바뀐다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data_loader import build_long_table, load_calendar
from .models import seasonal_naive_forecast


def build_forecast_artifact(data_dir: str = "data", train_end_d: str = "d_1913",
                             horizon: int = 28) -> dict:
    """id(item_store)별 예측치 + 최근 실적 + 가격/이벤트/SNAP 근거를 딕셔너리로 만든다."""
    long_df = build_long_table(data_dir)
    cal = load_calendar(data_dir)
    train_end = cal.loc[cal["d"] == train_end_d, "date"].iloc[0]

    forecast = seasonal_naive_forecast(long_df, train_end, horizon=horizon)
    forecast_dates = list(forecast.columns)

    hist = long_df[long_df["date"] <= train_end]
    recent_28 = hist[hist["date"] > train_end - pd.Timedelta(days=28)]
    recent_wide = recent_28.pivot(index="id", columns="date", values="sales").sort_index(axis=1)

    future_window = long_df[
        (long_df["date"] > train_end) & (long_df["date"] <= train_end + pd.Timedelta(days=horizon))
    ]
    events_in_window = (
        future_window[future_window["event_name_1"] != "none"]
        [["date", "event_name_1", "event_type_1"]]
        .drop_duplicates()
        .assign(date=lambda d: d["date"].dt.strftime("%Y-%m-%d"))
        .to_dict("records")
    )
    snap_days: dict[str, list[str]] = {}
    for state in ["CA", "TX", "WI"]:
        rows = future_window[(future_window["state_id"] == state) & (future_window["snap"] == 1)]
        snap_days[state] = sorted(rows["date"].drop_duplicates().dt.strftime("%Y-%m-%d").tolist())

    id_attrs = hist.drop_duplicates("id").set_index("id")[
        ["item_id", "store_id", "cat_id", "dept_id", "state_id"]
    ]
    latest_price = hist.sort_values("date").drop_duplicates("id", keep="last").set_index("id")["sell_price"]

    series: dict[str, dict] = {}
    for sid in forecast.index:
        attrs = id_attrs.loc[sid]
        series[sid] = {
            "item_id": attrs["item_id"], "store_id": attrs["store_id"], "cat_id": attrs["cat_id"],
            "dept_id": attrs["dept_id"], "state_id": attrs["state_id"],
            "forecast": {d.strftime("%Y-%m-%d"): float(forecast.loc[sid, d]) for d in forecast_dates},
            "recent_actual": {
                d.strftime("%Y-%m-%d"): float(recent_wide.loc[sid, d]) if sid in recent_wide.index else None
                for d in recent_wide.columns
            } if sid in recent_wide.index else {},
            "current_price": float(latest_price.get(sid, float("nan"))),
        }

    artifact = {
        "model": "seasonal_naive_7day",
        "train_end": train_end.strftime("%Y-%m-%d"),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in forecast_dates],
        "events_in_forecast_window": events_in_window,
        "snap_days_by_state": snap_days,
        "series": series,
    }
    return artifact


def save_artifact(artifact: dict, path: str = "artifacts/forecast_store.json") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(artifact, ensure_ascii=False), encoding="utf-8")


def load_artifact(path: str = "artifacts/forecast_store.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
