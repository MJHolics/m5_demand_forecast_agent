"""에이전트가 호출하는 도구. 전부 순수 함수 — LLM은 이 함수들이 돌려준 숫자만 근거로 답해야 한다
(Inspection Copilot에서 검증한 "도구가 진실, LLM은 서술" 패턴 재사용).
"""
from __future__ import annotations

MAX_MATCHED_SERIES = 30490  # 전체 시리즈 수(item_store 레벨)와 같게 둬서, 필터가 넓어도
# 합계가 조용히 잘리는 일이 없게 한다. 실제 상한이 필요해지면(예: 응답 문자열 폭주) 여기서 다시
# 낮추되, 그때는 잘렸다는 사실 자체를 결과에 명시해야 한다 — 잘렸는데 티 안 나는 게 진짜 사고다.


def find_series(artifact: dict, state_id: str | None = None, cat_id: str | None = None,
                 dept_id: str | None = None, store_id: str | None = None,
                 item_id: str | None = None) -> list[str]:
    """조건에 맞는 시리즈 id 목록. 조건을 안 주면(전부 None) 빈 리스트를 돌려준다 —
    "전체"를 의도한 질문은 find_series가 아니라 artifact 레벨 집계로 따로 다뤄야 한다."""
    filters = {"state_id": state_id, "cat_id": cat_id, "dept_id": dept_id,
               "store_id": store_id, "item_id": item_id}
    active = {k: v for k, v in filters.items() if v}
    if not active:
        return []
    matched = []
    for sid, s in artifact["series"].items():
        if all(s.get(k) == v for k, v in active.items()):
            matched.append(sid)
    return matched[:MAX_MATCHED_SERIES]


def get_forecast(artifact: dict, series_ids: list[str]) -> dict[str, float]:
    """주어진 시리즈들의 날짜별 예측 합계."""
    totals: dict[str, float] = {d: 0.0 for d in artifact["forecast_dates"]}
    for sid in series_ids:
        s = artifact["series"].get(sid)
        if not s:
            continue
        for d, v in s["forecast"].items():
            totals[d] = totals.get(d, 0.0) + v
    return totals


def get_recent_trend(artifact: dict, series_ids: list[str]) -> dict[str, float]:
    """주어진 시리즈들의 최근 28일 실제 판매량 합계(날짜별)."""
    totals: dict[str, float] = {}
    for sid in series_ids:
        s = artifact["series"].get(sid)
        if not s:
            continue
        for d, v in s.get("recent_actual", {}).items():
            if v is None:
                continue
            totals[d] = totals.get(d, 0.0) + v
    return dict(sorted(totals.items()))


def get_context(artifact: dict, series_ids: list[str]) -> dict:
    """가격(평균)·예측 구간 내 이벤트·관련 주(state)의 SNAP일 — 예측 근거로 쓰는 부가정보."""
    prices = [artifact["series"][sid]["current_price"] for sid in series_ids if sid in artifact["series"]]
    avg_price = sum(prices) / len(prices) if prices else None

    states = {artifact["series"][sid]["state_id"] for sid in series_ids if sid in artifact["series"]}
    snap_days = {st: artifact["snap_days_by_state"].get(st, []) for st in states}

    return {
        "avg_current_price": avg_price,
        "series_count": len(series_ids),
        "events_in_forecast_window": artifact["events_in_forecast_window"],
        "snap_days_by_state": snap_days,
        "model": artifact["model"],
    }
