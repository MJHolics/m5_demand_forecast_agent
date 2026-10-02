"""LLM 서술 단계가 도구 숫자를 지키는가 — 답변 속 숫자를 전부 뽑아 근거와 대조한다.

"도구가 진실, LLM은 서술" 구조라도 서술 단계에서 LLM이 합계·평균·변화율을 직접 계산하면
틀린 숫자가 근거처럼 나간다. 그걸 세는 벤치다.

판정 4가지(답변 1건 기준, 하나라도 걸리면 FAIL):
  grounded   답변의 숫자·날짜가 전부 도구 결과(그대로 또는 합계·평균·최대/최소·주별 합·차이·변화율)로
             설명되는가 — 적힌 자릿수의 반 단위 이내 (`m5fc/narration_guard.py`, 운영 게이트와 같은 함수)
  key        질문 유형별 정답값이 들어 있는가(합계를 물으면 합계) — dev v1 답변을 읽고 추가했다.
             숫자 근거율만으로는 "4주 합계를 물었는데 4일치만 나열"이 통과했다. open 유형은 판정 안 함
  clean      출력 한도 잘림·한국어 외 문자 유출·증감 방향 모순이 없는가
v3(게이트+템플릿 폴백)는 폴백된 답을 따로 센다: 틀린 답이 나간 것과 안전하게 물러난 것은 다르다.

사용(로컬 vLLM 기동 후):
  LOCAL_LLM_BASE_URL=http://localhost:8000/v1 python tools/bench_narration.py --prompt v1 --tag qwen7b-awq
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from m5fc.agent import build_graph  # noqa: E402
from m5fc.agent_llm import LLM  # noqa: E402
from m5fc.forecast_store import load_artifact  # noqa: E402
from m5fc.narration_guard import (NUM, _DATE_ISO, check_numbers, direction_conflict,  # noqa: E402
                                  foreign_leak, matches)
from narration_queries import DEV_IDX, QTYPES, QUERIES, TEST_IDX  # noqa: E402

OUT = ROOT / "results" / "narration"

_UP = ("증가", "늘", "상승", "오를", "높아", "많아")
_DOWN = ("감소", "줄", "하락", "낮아", "적어")
_EVENTS = ("pesach", "orthodox", "cinco", "mother")


def _has(answer: str, targets: set[float]) -> bool:
    """답변에 targets 중 하나가 적힌 자릿수 기준으로 들어 있는가(만·천 단위 포함)."""
    for m in NUM.finditer(_DATE_ISO.sub(" ", answer)):
        mult = {"만": 1e4, "천": 1e3}.get(m.group(2), 1)
        if matches(m.group(1), mult, targets):
            return True
    return False


def _direction_ok(answer: str, up: bool) -> bool:
    right, wrong = (_UP, _DOWN) if up else (_DOWN, _UP)
    return any(w in answer for w in right) and not any(w in answer for w in wrong)


def check_key(answer: str, state: dict, qtype: str) -> bool | None:
    """질문 유형별 정답값이 답변에 있는가. open이면 None(판정 안 함)."""
    if qtype == "open":
        return None
    fc = list(state["forecast"].values())
    rc = list(state["recent_trend"].values())
    sf, sr = sum(fc), sum(rc)
    low = answer.lower()
    checks = []
    for t in qtype.split("+"):
        if t == "total":
            checks.append(_has(answer, {sf}))
        elif t == "avg":
            checks.append(_has(answer, {sf / len(fc)}))
        elif t == "recent_total":
            checks.append(_has(answer, {sr}))
        elif t == "weekly":
            checks.append(_has(answer, {sum(fc[i:i + 7]) for i in range(0, len(fc), 7)}))
        elif t == "pct":
            pct = abs(sf / sr - 1) * 100
            checks.append(_has(answer, {pct}) and _direction_ok(answer, sf > sr))
        elif t == "diff":
            checks.append(_has(answer, {abs(sf - sr)}) or _has(answer, {abs(sf / sr - 1) * 100}))
        elif t == "compare":
            checks.append(_direction_ok(answer, sf > sr))
        elif t in ("max", "min"):
            target = max(fc) if t == "max" else min(fc)
            days = {d for d, v in state["forecast"].items() if v == target}
            md = {f"{int(d[5:7])}월 {int(d[8:10])}일" for d in days} | days
            checks.append(_has(answer, {target}) or any(x in answer for x in md))
        elif t == "event":
            checks.append(all(e in low for e in _EVENTS))
    return all(checks)


def run(prompt: str, tag: str, split: str) -> dict:
    artifact = load_artifact(str(ROOT / "artifacts" / "forecast_store.json"))
    llm = LLM()
    calls: list[dict] = []

    def complete(system: str, user: str) -> str:
        t0 = time.perf_counter()
        out = llm.complete(system, user)
        calls.append({"latency_s": time.perf_counter() - t0, "prompt_chars": len(system) + len(user),
                      "finish_reason": getattr(llm, "last_finish_reason", None), "raw": out})
        return out

    app = build_graph(artifact, complete, prompt_version=prompt)
    complete("ping", "1+1=?")  # 콜드 스타트가 첫 질의 지연을 오염시키지 않게 워밍업
    calls.clear()

    idx = {"dev": DEV_IDX, "test": TEST_IDX, "all": list(range(len(QUERIES)))}[split]
    rows = []
    for i in idx:
        st = app.invoke({"query": QUERIES[i]})
        ans = st["answer"]
        call = calls[-1]
        fallback = bool(st.get("guard_reasons"))
        chk = check_numbers(ans, st)
        key = check_key(ans, st, QTYPES[i])
        truncated = call["finish_reason"] == "length" and not fallback
        leak = foreign_leak(ans)
        conflict = direction_conflict(ans, st)
        passed = chk["grounded"] and key is not False and not truncated and not leak and not conflict
        rows.append({"i": i, "split": "dev" if i in DEV_IDX else "test", "query": QUERIES[i],
                     "qtype": QTYPES[i], "answer": ans, **chk, "key_correct": key,
                     "truncated": truncated, "foreign_leak": leak, "direction_conflict": conflict,
                     "fallback": fallback, "guard_reasons": st.get("guard_reasons", []),
                     "pass": passed, **call})
        print(f"[{i:2d}] {'PASS' if passed else ('FALLBACK' if fallback else 'FAIL')} "
              f"grounded={chk['grounded']} key={key} trunc={truncated} leak={leak} conflict={conflict} "
              f"bad={chk['unsupported'][:3]} guard={st.get('guard_reasons', [])}", flush=True)
    n = len(rows)
    keyed = [r for r in rows if r["key_correct"] is not None]
    summary = {
        "prompt": prompt, "model": llm.model, "split": split, "n": n,
        "pass_rate": sum(r["pass"] for r in rows) / n,
        "wrong_delivered": sum(not r["pass"] and not r["fallback"] for r in rows),
        "fallback": sum(r["fallback"] for r in rows),
        "grounded_rate": sum(r["grounded"] for r in rows) / n,
        "key_correct_rate": sum(r["key_correct"] for r in keyed) / len(keyed) if keyed else None,
        "n_keyed": len(keyed),
        "truncated": sum(r["truncated"] for r in rows),
        "foreign_leak": sum(r["foreign_leak"] for r in rows),
        "direction_conflict": sum(r["direction_conflict"] for r in rows),
        "numbers_total": sum(r["n_numbers"] for r in rows),
        "unsupported_total": sum(len(r["unsupported"]) for r in rows),
        "latency_p50_s": sorted(r["latency_s"] for r in rows)[n // 2],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"narration_{prompt}_{tag}_{split}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="v1")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    a = ap.parse_args()
    run(a.prompt, a.tag, a.split)
