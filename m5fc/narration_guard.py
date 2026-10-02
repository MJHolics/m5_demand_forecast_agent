"""LLM 서술 검증 게이트 — 답변의 숫자·날짜·방향을 도구 결과와 대조한다.

벤치(`tools/bench_narration.py`)와 운영 경로(`agent.py`의 v3)가 같은 판정 함수를 쓴다.
판정은 정규식 기반이라 의미 오류(근거 없는 인과 주장 등)는 못 잡는다 — 잡는 것은
① 도구 결과로 설명되지 않는 숫자·날짜 ② 도구가 낸 증감 방향과 반대되는 서술 ③ 한국어 외 문자 유출(템플릿 붕괴).
"""
from __future__ import annotations

import re

_DATE_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_DATE_MD = re.compile(r"\b(0[1-9]|1[0-2])-([0-3]\d)\b")
_DATE_KO = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_DAY_KO = re.compile(r"(\d{1,2})\s*일(?!간|치|\s*동안)")  # "5월 1일부터 8일까지"의 8일
_YEAR_KO = re.compile(r"\d{4}\s*년")
_MONTH_KO = re.compile(r"\d{1,2}\s*월")
_ID = re.compile(r"\b[A-Za-z]+_\d+\b")
NUM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(만|천|%|퍼센트)?")
_FOREIGN = re.compile(r"[一-鿿Ѐ-ӿ]")  # 한자·키릴 — 템플릿 붕괴 때 섞여 나왔다
_UP_STRICT = ("증가", "늘어", "늘 ", "상승")
_DOWN_STRICT = ("감소", "줄어", "하락")


def allowed_values(state: dict) -> tuple[set[float], set[float], set[tuple[int, int]]]:
    """(일반 숫자 허용 집합, 퍼센트 허용 집합, 허용 날짜(월,일) 집합)."""
    fc = list(state["forecast"].values())
    rc = list(state["recent_trend"].values())
    ctx = state["context"]
    vals: set[float] = set(fc) | set(rc)
    for xs in (fc, rc):
        if not xs:
            continue
        vals |= {sum(xs), sum(xs) / len(xs), max(xs), min(xs)}
        vals |= {sum(xs[i:i + 7]) for i in range(0, len(xs), 7)}
        vals |= {sum(xs[i:i + 7]) / 7 for i in range(0, len(xs), 7)}
    sf, sr = sum(fc), sum(rc)
    vals |= {abs(sf - sr), ctx["series_count"], 28, 7, 4, 1, 2, 3,
             len(ctx["events_in_forecast_window"]),
             *[len(v) for v in ctx["snap_days_by_state"].values()]}
    if ctx["avg_current_price"] is not None:
        vals.add(ctx["avg_current_price"])
    pcts: set[float] = set()
    if sr:
        pcts |= {abs(sf / sr - 1) * 100, sf / sr * 100}
    dates = set(state["forecast"]) | set(state["recent_trend"])
    dates |= {e["date"] for e in ctx["events_in_forecast_window"]}
    for v in ctx["snap_days_by_state"].values():
        dates |= set(v)
    md = {(int(d[5:7]), int(d[8:10])) for d in dates}
    return vals, pcts, md


def matches(written: str, mult: float, targets: set[float]) -> bool:
    """적힌 자릿수의 반 단위 이내면 일치("12.7만"은 ±500, "126,584"는 ±0.5)."""
    w = float(written.replace(",", "")) * mult
    dec = len(written.split(".")[1]) if "." in written else 0
    tol = 0.5 * 10 ** (-dec) * mult + 1e-9
    return any(abs(w - t) <= tol for t in targets)


def check_numbers(answer: str, state: dict) -> dict:
    vals, pcts, md = allowed_values(state)
    days = {d for _, d in md}
    bad: list[str] = []
    text = answer
    for rx in (_DATE_ISO, _DATE_MD, _DATE_KO):
        for m in rx.finditer(text):
            mm, dd = (m.group(2), m.group(3)) if rx is _DATE_ISO else (m.group(1), m.group(2))
            if (int(mm), int(dd)) not in md:
                bad.append(m.group(0))
        text = rx.sub(" ", text)
    for m in _DAY_KO.finditer(text):
        if int(m.group(1)) not in days and int(m.group(1)) != 28:
            bad.append(m.group(0))
    text = _DAY_KO.sub(" ", text)
    for rx in (_YEAR_KO, _MONTH_KO, _ID):
        text = rx.sub(" ", text)
    n_numbers = 0
    for m in NUM.finditer(text):
        num, unit = m.group(1), m.group(2)
        n_numbers += 1
        if unit in ("%", "퍼센트"):
            ok = matches(num, 1, pcts)
        else:
            ok = matches(num, {"만": 1e4, "천": 1e3}.get(unit, 1), vals)
        if not ok:
            bad.append(m.group(0).strip())
    return {"n_numbers": n_numbers, "unsupported": bad, "grounded": not bad}


def direction_conflict(answer: str, state: dict) -> bool:
    """도구가 낸 증감 방향과 반대 단어가 답에 있으면 True. 최근 실적이 없으면 판정 안 함."""
    sf, sr = sum(state["forecast"].values()), sum(state["recent_trend"].values())
    if not sr or sf == sr:
        return False
    wrong = _DOWN_STRICT if sf > sr else _UP_STRICT
    return any(w in answer for w in wrong)


def foreign_leak(answer: str) -> bool:
    return bool(_FOREIGN.search(answer))


def validate(answer: str, state: dict) -> tuple[bool, list[str]]:
    """운영 게이트. (통과 여부, 걸린 이유 목록)."""
    reasons = []
    chk = check_numbers(answer, state)
    if not chk["grounded"]:
        reasons.append(f"unsupported:{chk['unsupported'][:3]}")
    if direction_conflict(answer, state):
        reasons.append("direction_conflict")
    if foreign_leak(answer):
        reasons.append("foreign_leak")
    if not answer.strip():
        reasons.append("empty")
    return not reasons, reasons
