"""서술 검증 게이트·집계 도구 — 벤치(tools/bench_narration.py)가 실측에서 본 실패를 고정한다."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m5fc.agent import build_graph
from m5fc.agent_tools import summarize_forecast
from m5fc.narration_guard import check_numbers, direction_conflict, foreign_leak, validate


def _state(fc_vals, rc_vals):
    fc = {f"2016-04-{25 + i:02d}" if i < 6 else f"2016-05-{i - 5:02d}": v for i, v in enumerate(fc_vals)}
    rc = {f"2016-04-{18 + i:02d}": v for i, v in enumerate(rc_vals)}
    return {"forecast": fc, "recent_trend": rc,
            "context": {"series_count": 3, "avg_current_price": 2.5, "model": "m",
                        "events_in_forecast_window": [{"date": "2016-05-01", "event_name_1": "OrthodoxEaster"}],
                        "snap_days_by_state": {"CA": ["2016-05-01", "2016-05-02"]}}}


def test_summary_precomputes_what_v1_got_wrong():
    s = summarize_forecast({"2016-04-25": 10.0, "2016-04-26": 30.0}, {"2016-04-18": 20.0, "2016-04-19": 20.0})
    assert s["forecast_total_28d"] == 40.0
    assert s["forecast_daily_avg"] == 20.0
    assert s["peak_days"] == ["2016-04-26"] and s["low_value"] == 10.0
    assert s["change_vs_recent"] == {"direction": "변화 없음", "pct": 0.0, "abs_diff": 0.0}


def test_partial_sum_is_rejected():
    # v1 dev #0: "4주 총량"을 4일치만 더해 답했다 — 합계 집합에 없는 숫자
    st = _state([100.0] * 7, [90.0] * 7)
    assert check_numbers("총 700개입니다", st)["grounded"]
    assert not check_numbers("총 400개입니다", st)["grounded"]


def test_inflated_percentage_is_rejected():
    # v1 dev #6: 1.3%를 계산해 놓고 29.1%로 바꿔 썼다
    st = _state([101.0] * 7, [100.0] * 7)
    assert check_numbers("1.0% 증가", st)["grounded"]
    assert not check_numbers("29.1% 증가", st)["grounded"]


def test_out_of_window_date_is_rejected():
    st = _state([1.0] * 7, [1.0] * 7)
    assert check_numbers("5월 1일이 최대", st)["grounded"]
    assert not check_numbers("6월 30일이 최대", st)["grounded"]


def test_direction_conflict_and_foreign_leak():
    st = _state([90.0] * 7, [100.0] * 7)  # 감소
    assert direction_conflict("수요가 증가할 것", st)
    assert not direction_conflict("10.0% 감소", st)
    assert foreign_leak("이 값은 预测 부분")
    assert not foreign_leak("Mother's day 수요")


def test_v3_falls_back_to_template_when_gate_fails():
    from test_agent import _toy_artifact  # 기존 그래프 테스트의 소형 아티팩트 재사용

    artifact = _toy_artifact()
    bad_llm = lambda system, user: "총 999,999개 판매, 预测"  # noqa: E731
    out = build_graph(artifact, bad_llm, prompt_version="v3").invoke({"query": "캘리포니아 FOODS 수요는?"})
    assert out["guard_reasons"]
    assert "999,999" not in out["answer"]
    ok, _ = validate(out["answer"], out)
    assert ok
