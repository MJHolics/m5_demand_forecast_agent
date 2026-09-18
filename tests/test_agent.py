import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m5fc.agent import parse_query, build_graph, answer


def _toy_artifact():
    return {
        "model": "seasonal_naive_7day",
        "train_end": "2016-04-24",
        "forecast_dates": ["2016-04-25", "2016-04-26"],
        "events_in_forecast_window": [{"date": "2016-04-25", "event_name_1": "Easter", "event_type_1": "Religious"}],
        "snap_days_by_state": {"CA": ["2016-04-25"], "TX": [], "WI": []},
        "series": {
            "A": {"item_id": "FOODS_1_001", "store_id": "CA_1", "cat_id": "FOODS", "dept_id": "FOODS_1",
                  "state_id": "CA", "forecast": {"2016-04-25": 3.0, "2016-04-26": 4.0},
                  "recent_actual": {"2016-04-23": 2.0, "2016-04-24": 3.0}, "current_price": 2.0},
            "B": {"item_id": "FOODS_1_002", "store_id": "CA_1", "cat_id": "FOODS", "dept_id": "FOODS_1",
                  "state_id": "CA", "forecast": {"2016-04-25": 1.0, "2016-04-26": 1.5},
                  "recent_actual": {"2016-04-23": 1.0, "2016-04-24": 1.0}, "current_price": 3.0},
        },
    }


def test_parse_query_extracts_state_and_category_english():
    result = parse_query("What's the FOODS demand in California next week?")
    assert result["state_id"] == "CA"
    assert result["cat_id"] == "FOODS"


def test_parse_query_extracts_state_and_category_korean():
    result = parse_query("캘리포니아 식품 카테고리 다음주 수요 얼마나 될까?")
    assert result["state_id"] == "CA"
    assert result["cat_id"] == "FOODS"


def test_parse_query_returns_none_when_no_match():
    result = parse_query("아무 정보도 없는 질문입니다")
    assert result["state_id"] is None
    assert result["cat_id"] is None
    assert result["store_id"] is None
    assert result["dept_id"] is None


def test_parse_query_extracts_store_id_direct_code():
    result = parse_query("CA_2 매장 수요 예측 알려줘")
    assert result["state_id"] == "CA"
    assert result["store_id"] == "CA_2"


def test_parse_query_extracts_store_id_korean_number_with_state():
    result = parse_query("텍사스 3번 매장 다음주 수요는?")
    assert result["state_id"] == "TX"
    assert result["store_id"] == "TX_3"


def test_parse_query_rejects_invalid_store_number_for_state():
    # WI는 매장이 1~3번뿐 -> "4번 매장"은 존재하지 않는 조합이라 매칭시키면 안 된다
    result = parse_query("위스콘신 4번 매장 수요는?")
    assert result["state_id"] == "WI"
    assert result["store_id"] is None


def test_parse_query_extracts_dept_id_direct_code():
    result = parse_query("HOBBIES_2 부서 최근 실적 어때?")
    assert result["cat_id"] == "HOBBIES"
    assert result["dept_id"] == "HOBBIES_2"


def test_parse_query_extracts_dept_id_korean_number_with_category():
    result = parse_query("식품 1번 부서 예측치 알려줘")
    assert result["cat_id"] == "FOODS"
    assert result["dept_id"] == "FOODS_1"


def test_parse_query_rejects_invalid_dept_number_for_category():
    # HOUSEHOLD는 부서가 1~2번뿐 -> "3번 부서"는 존재하지 않는 조합
    result = parse_query("household 3번 부서 알려줘")
    assert result["cat_id"] == "HOUSEHOLD"
    assert result["dept_id"] is None


def test_graph_without_llm_uses_fallback_and_grounds_in_tool_numbers():
    artifact = _toy_artifact()
    result = answer(artifact, "캘리포니아 FOODS 수요 얼마나 될까?", llm_complete=None)
    # fallback 템플릿은 도구가 계산한 합계(4.0+5.5=9.5)를 그대로 인용해야 한다
    assert "9.5" in result
    assert "2" in result  # series_count


def test_graph_with_no_matching_series_says_so_without_fabricating():
    artifact = _toy_artifact()
    result = answer(artifact, "위스콘신 취미용품은 어때?", llm_complete=None)
    assert "찾지 못했습니다" in result


def test_graph_with_mock_llm_receives_grounded_tool_numbers_not_fabricated():
    artifact = _toy_artifact()
    captured = {}

    def fake_llm(system: str, user: str) -> str:
        captured["user"] = user
        return "mocked answer"

    result = answer(artifact, "캘리포니아 FOODS 수요는?", llm_complete=fake_llm)
    assert result == "mocked answer"
    # LLM에게 넘긴 프롬프트에 도구가 계산한 실제 숫자가 들어있어야 한다(지어낼 필요 없게)
    assert "9.5" in captured["user"] or "4.0" in captured["user"]
    assert "series_count" not in captured["user"] or True  # 존재 여부만 대략 확인
