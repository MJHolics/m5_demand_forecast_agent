"""제공자 무관 LLM 클라이언트 — Gemini(무료 티어) 기본, Anthropic·OpenAI 옵션.

`inspection-copilot/app/llm.py`와 동일한 패턴을 재사용한다(이 프로젝트 시리즈 전체의 관행).
키가 없으면 호출 시점에 RuntimeError를 던진다 — `agent.py`는 이 경우 결정론적 템플릿으로
대체하므로, 키 없는 환경에서도 도구 체인 자체는 전부 테스트 가능하다.
"""
from __future__ import annotations

import os


def _detect_provider() -> str:
    provider = os.getenv("LLM_PROVIDER", "auto").lower()
    if provider != "auto":
        return provider
    if os.getenv("LOCAL_LLM_BASE_URL"):
        return "local"
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError(
        "LLM API 키가 없습니다. GEMINI_API_KEY(무료: aistudio.google.com) 또는 "
        "ANTHROPIC_API_KEY / OPENAI_API_KEY 중 하나를 설정하세요."
    )


DEFAULT_MODELS = {
    "gemini": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
    "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    "local": os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
}


class LLM:
    def __init__(self) -> None:
        self.provider = _detect_provider()
        self.model = DEFAULT_MODELS[self.provider]
        self._client = None

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        fn = {"gemini": self._gemini, "anthropic": self._anthropic, "openai": self._openai,
              "local": self._local}[self.provider]
        return fn(system, user, temperature)

    def _gemini(self, system: str, user: str, temperature: float) -> str:
        if self._client is None:
            from google import genai
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            self._client = genai.Client(api_key=key)
        from google.genai import types
        resp = self._client.models.generate_content(
            model=self.model, contents=user,
            config=types.GenerateContentConfig(system_instruction=system, temperature=temperature),
        )
        return (resp.text or "").strip()

    def _anthropic(self, system: str, user: str, temperature: float) -> str:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        msg = self._client.messages.create(
            model=self.model, max_tokens=1024, temperature=temperature,
            system=system, messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()

    def _openai(self, system: str, user: str, temperature: float) -> str:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = self._client.chat.completions.create(
            model=self.model, temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()

    def _local(self, system: str, user: str, temperature: float) -> str:
        """온프레미스 OpenAI 호환 서버(vLLM 등). 키 불요 — Inspection Copilot `_local`과 같은 형태."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=os.getenv("LOCAL_LLM_BASE_URL"), api_key="EMPTY", timeout=120)
        resp = self._client.chat.completions.create(
            model=self.model, temperature=temperature, max_tokens=512,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        self.last_finish_reason = resp.choices[0].finish_reason  # "length"면 출력 한도에서 잘린 답
        return (resp.choices[0].message.content or "").strip()


def make_llm_complete():
    """agent.py의 llm_complete(system, user) -> str 시그니처에 맞춘 팩토리.
    키가 없으면 None을 돌려줘서 agent.py가 자동으로 템플릿 폴백을 쓰게 한다."""
    try:
        llm = LLM()
    except RuntimeError:
        return None
    return lambda system, user: llm.complete(system, user)
