"""자연어 질의 → 예측 조회 → 근거 조회 → 답변 생성. LangGraph StateGraph로 결선.

라우팅(질의에서 조건 추출)은 규칙 기반이다 — Inspection Copilot의 RuleRouter와 같은 이유:
자유 발화 라우팅을 LLM에 맡기면 재현이 안 되고 테스트도 못 한다. LLM은 도구가 이미 구한 숫자를
문장으로 옮기는 마지막 단계에서만 쓴다("도구가 진실, LLM은 서술").
"""
from __future__ import annotations

import re
from typing import Callable, TypedDict

from .agent_tools import find_series, get_context, get_forecast, get_recent_trend, summarize_forecast
from .narration_guard import validate

STATE_KEYWORDS = {
    "CA": ["ca", "california", "캘리포니아"],
    "TX": ["tx", "texas", "텍사스"],
    "WI": ["wi", "wisconsin", "위스콘신"],
}
CAT_KEYWORDS = {
    "FOODS": ["foods", "food", "식품"],
    "HOBBIES": ["hobbies", "hobby", "취미"],
    "HOUSEHOLD": ["household", "생활용품", "가정용품"],
}

# 실데이터로 확인한 유효 조합만 인정 — M5는 주 4/3/3개 매장, 카테고리당 부서 1~3개로 조합이
# 정해져 있다(store_id: sales_train_evaluation.csv 실측, dept_id 동일). 여기 없는 조합은
# "3번 매장"처럼 숫자만 맞아도 매칭시키지 않는다 — 존재하지 않는 시리즈를 지어내는 게 더 나쁘다.
VALID_STORES = {"CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"}
VALID_DEPTS = {"FOODS_1", "FOODS_2", "FOODS_3", "HOBBIES_1", "HOBBIES_2", "HOUSEHOLD_1", "HOUSEHOLD_2"}

_STORE_CODE_RE = re.compile(r"\b(ca|tx|wi)[_\s-]?([1-4])\b", re.IGNORECASE)
_STORE_KOREAN_RE = re.compile(r"([1-9])\s*번\s*매장")
_DEPT_CODE_RE = re.compile(r"\b(foods|hobbies|household)[_\s-]?([1-3])\b", re.IGNORECASE)
_DEPT_KOREAN_RE = re.compile(r"([1-9])\s*번?\s*부서")


def _find_store_id(text: str, state_id: str | None) -> str | None:
    """자유 발화에서 매장 코드 추출. "CA_1"류 직접 코드 우선, 그다음 "N번 매장"+주 조합."""
    m = _STORE_CODE_RE.search(text)
    if m:
        candidate = f"{m.group(1).upper()}_{m.group(2)}"
        if candidate in VALID_STORES:
            return candidate
    if state_id:
        m = _STORE_KOREAN_RE.search(text)
        if m:
            candidate = f"{state_id}_{m.group(1)}"
            if candidate in VALID_STORES:
                return candidate
    return None


def _find_dept_id(text: str, cat_id: str | None) -> str | None:
    """자유 발화에서 부서 코드 추출. "FOODS_1"류 직접 코드 우선, 그다음 "N번 부서"+카테고리 조합."""
    m = _DEPT_CODE_RE.search(text)
    if m:
        candidate = f"{m.group(1).upper()}_{m.group(2)}"
        if candidate in VALID_DEPTS:
            return candidate
    if cat_id:
        m = _DEPT_KOREAN_RE.search(text)
        if m:
            candidate = f"{cat_id}_{m.group(1)}"
            if candidate in VALID_DEPTS:
                return candidate
    return None


def parse_query(text: str) -> dict:
    """질의 텍스트에서 state/category/store/dept 조건을 규칙 기반으로 뽑는다. 못 찾으면 None.
    (item_id 직접 지정은 아직 정확한 코드로만 매칭 — 품목 3,049개는 자유 발화 사전이 너무 커서
    다음 단계로 남긴다)"""
    lower = text.lower()
    state_id = next((k for k, kws in STATE_KEYWORDS.items() if any(kw in lower for kw in kws)), None)
    cat_id = next((k for k, kws in CAT_KEYWORDS.items() if any(kw in lower for kw in kws)), None)
    store_id = _find_store_id(text, state_id)
    dept_id = _find_dept_id(text, cat_id)
    return {"state_id": state_id, "cat_id": cat_id, "store_id": store_id, "dept_id": dept_id}


class AgentState(TypedDict, total=False):
    query: str
    filters: dict
    series_ids: list[str]
    forecast: dict[str, float]
    recent_trend: dict[str, float]
    context: dict
    answer: str
    guard_reasons: list[str]


def node_parse(state: AgentState) -> AgentState:
    return {"filters": parse_query(state["query"])}


def make_node_lookup(artifact: dict) -> Callable[[AgentState], AgentState]:
    def node_lookup(state: AgentState) -> AgentState:
        series_ids = find_series(artifact, **state["filters"])
        return {"series_ids": series_ids}
    return node_lookup


def make_node_tools(artifact: dict) -> Callable[[AgentState], AgentState]:
    def node_tools(state: AgentState) -> AgentState:
        series_ids = state["series_ids"]
        if not series_ids:
            return {"forecast": {}, "recent_trend": {}, "context": {}}
        return {
            "forecast": get_forecast(artifact, series_ids),
            "recent_trend": get_recent_trend(artifact, series_ids),
            "context": get_context(artifact, series_ids),
        }
    return node_tools


def _fallback_synthesize(state: AgentState) -> str:
    """LLM 없이도 동작하는 템플릿 답변 — 키 없는 환경에서도 도구 체인은 검증 가능해야 한다."""
    if not state["series_ids"]:
        return "조건에 맞는 시리즈를 찾지 못했습니다. 주(CA/TX/WI)나 카테고리(FOODS/HOBBIES/HOUSEHOLD)를 포함해 다시 물어봐 주세요."
    forecast = state["forecast"]
    total = sum(forecast.values())
    n = state["context"]["series_count"]
    events = state["context"]["events_in_forecast_window"]
    event_note = f" 예측 기간 중 이벤트: {', '.join(e['event_name_1'] for e in events)}." if events else ""
    return (
        f"조건에 맞는 {n}개 시리즈의 예측 총합은 {total:.1f}개입니다 "
        f"(모델: {state['context']['model']}, 평균 현재가 {state['context']['avg_current_price']:.2f}).{event_note}"
    )


def make_node_synthesize(llm_complete: Callable[[str, str], str] | None = None,
                         prompt_version: str = "v1") -> Callable[[AgentState], AgentState]:
    """llm_complete(system, user) -> str 형태의 콜러블을 주면 LLM으로 서술하고,
    안 주면(키 없는 환경 등) 결정론적 템플릿으로 대체한다."""
    def node_synthesize(state: AgentState) -> AgentState:
        if llm_complete is None:
            return {"answer": _fallback_synthesize(state)}
        if not state["series_ids"]:
            return {"answer": _fallback_synthesize(state)}
        if prompt_version == "v3":
            # v2 프롬프트 + 검증 게이트: 표로 설명 안 되는 숫자·방향 모순·외국 문자가 있으면
            # LLM 답을 버리고 결정론적 템플릿으로 물러난다(틀린 답보다 덜 친절한 답이 낫다).
            system, user = _prompt_v2(state)
            draft = llm_complete(system, user)
            ok, reasons = validate(draft, state)
            if ok:
                return {"answer": draft, "guard_reasons": []}
            return {"answer": _fallback_synthesize(state), "guard_reasons": reasons}
        if prompt_version != "v1":
            system, user = PROMPTS[prompt_version](state)
            return {"answer": llm_complete(system, user)}
        system = (
            "너는 소매 수요예측 결과를 설명하는 어시스턴트다. 아래 도구 결과에 있는 숫자만 근거로 "
            "답하고, 도구 결과에 없는 숫자는 절대 지어내지 마라."
        )
        user = (
            f"질문: {state['query']}\n"
            f"매칭된 시리즈 수: {state['context']['series_count']}\n"
            f"예측(날짜별 합계): {state['forecast']}\n"
            f"최근 28일 실적(날짜별 합계): {state['recent_trend']}\n"
            f"평균 현재가: {state['context']['avg_current_price']}\n"
            f"예측 기간 중 이벤트: {state['context']['events_in_forecast_window']}\n"
            f"관련 주 SNAP일: {state['context']['snap_days_by_state']}\n"
        )
        return {"answer": llm_complete(system, user)}
    return node_synthesize


_WEEKDAYS = "월화수목금토일"


def _fmt(x: float) -> str:
    return f"{x:,.0f}" if float(x).is_integer() else f"{x:,.1f}"


def _prompt_v2(state: AgentState) -> tuple[str, str]:
    """집계는 도구가 미리 내고, LLM에는 요약표만 준다(일별 원시값 28+28개 대신 첫 주 요일 패턴 7개).
    v1 dev에서 본 실패(직접 합산 오류·변화율 바꿔치기·나열하다 잘림)를 입력에서 없앤다."""
    from datetime import date

    s = summarize_forecast(state["forecast"], state["recent_trend"])
    ctx = state["context"]
    first_week = sorted(state["forecast"])[:7]
    pattern = ", ".join(
        f"{_WEEKDAYS[date.fromisoformat(d).weekday()]}({d[5:]}) {_fmt(state['forecast'][d])}" for d in first_week
    )
    ch = s["change_vs_recent"]
    change = (f"최근 28일 대비 {ch['direction']} {ch['pct']:.1f}% (차이 {_fmt(ch['abs_diff'])}개)"
              if ch else "최근 실적 없음")
    events = ", ".join(f"{e['date']} {e['event_name_1']}" for e in ctx["events_in_forecast_window"]) or "없음"
    snap = "; ".join(f"{st} {len(days)}일({', '.join(d[5:] for d in days)})"
                     for st, days in ctx["snap_days_by_state"].items())
    system = (
        "너는 소매 수요예측 결과를 설명하는 어시스턴트다. 한국어로만 3~5문장으로 답한다.\n"
        "규칙: 숫자는 아래 [요약표]에 적힌 값을 그대로 옮겨 쓴다. 더하기·나누기·변화율 같은 계산을 직접 하지 마라 — "
        "필요한 계산은 이미 표에 있다. 표에 없는 숫자가 필요한 질문이면 '제공된 결과에 없다'고 답한다. "
        "일별 값을 나열하지 말고 질문에 필요한 값만 쓴다."
    )
    user = (
        f"질문: {state['query']}\n\n[요약표]\n"
        f"- 예측 모델: {ctx['model']} (최근 1주 패턴을 4주 반복), 매칭 시리즈 {ctx['series_count']}개\n"
        f"- 예측 기간: {min(state['forecast'])} ~ {max(state['forecast'])} (28일)\n"
        f"- 4주 예측 합계: {_fmt(s['forecast_total_28d'])}개 · 하루 평균 {_fmt(s['forecast_daily_avg'])}개\n"
        f"- 주별 예측 합계: {' / '.join(_fmt(w) for w in s['forecast_weekly_totals'])}\n"
        f"- 최근 28일 실적 합계: {_fmt(s['recent_total_28d'])}개 → {change}\n"
        f"- 요일별 예측(첫 주): {pattern}\n"
        f"- 예측 최대: {_fmt(s['peak_value'])}개 ({', '.join(s['peak_days'])})\n"
        f"- 예측 최소: {_fmt(s['low_value'])}개 ({', '.join(s['low_days'])})\n"
        f"- 평균 현재가: {ctx['avg_current_price']:.2f}달러\n"
        f"- 예측 기간 이벤트: {events}\n"
        f"- SNAP일: {snap}\n"
    )
    return system, user


PROMPTS: dict[str, Callable[[AgentState], tuple[str, str]]] = {"v2": _prompt_v2}


def build_graph(artifact: dict, llm_complete: Callable[[str, str], str] | None = None,
                prompt_version: str = "v1"):
    from langgraph.graph import END, StateGraph

    graph = StateGraph(AgentState)
    graph.add_node("parse", node_parse)
    graph.add_node("lookup", make_node_lookup(artifact))
    graph.add_node("tools", make_node_tools(artifact))
    graph.add_node("synthesize", make_node_synthesize(llm_complete, prompt_version))

    graph.set_entry_point("parse")
    graph.add_edge("parse", "lookup")
    graph.add_edge("lookup", "tools")
    graph.add_edge("tools", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def answer(artifact: dict, query: str, llm_complete: Callable[[str, str], str] | None = None) -> str:
    app = build_graph(artifact, llm_complete)
    result = app.invoke({"query": query})
    return result["answer"]
