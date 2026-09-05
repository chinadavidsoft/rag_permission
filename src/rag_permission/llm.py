from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from openai import OpenAI


@dataclass(frozen=True, slots=True)
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    usage: LLMUsage = LLMUsage()


@dataclass(frozen=True, slots=True)
class TokenDelta:
    text: str


@dataclass(frozen=True, slots=True)
class UsageDelta:
    usage: LLMUsage


StreamDelta = TokenDelta | UsageDelta


class LLMClient(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse: ...

    def stream(
        self, system: str, user: str, temperature: float = 0.0
    ) -> Iterator[StreamDelta]: ...


class OpenAILLMClient:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    @staticmethod
    def _usage(raw_usage: Any) -> LLMUsage:
        if raw_usage is None:
            return LLMUsage()
        return LLMUsage(
            prompt_tokens=int(getattr(raw_usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw_usage, "completion_tokens", 0) or 0),
        )

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
        )
        return LLMResponse(
            text=response.choices[0].message.content or "",
            usage=self._usage(response.usage),
        )

    def stream(self, system: str, user: str, temperature: float = 0.0) -> Iterator[StreamDelta]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield TokenDelta(chunk.choices[0].delta.content)
            usage = self._usage(getattr(chunk, "usage", None))
            if usage.total_tokens:
                yield UsageDelta(usage)
