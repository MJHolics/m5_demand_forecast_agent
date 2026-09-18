import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from m5fc.agent_llm import make_llm_complete


def test_make_llm_complete_returns_none_without_any_api_key(monkeypatch):
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert make_llm_complete() is None
