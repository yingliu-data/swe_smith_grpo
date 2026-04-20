from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import NEBIUS_BASE_URL, NEBIUS_MODEL


@dataclass(slots=True)
class NebiusResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int


class NebiusClient:
    """Thin async wrapper around the Nebius OpenAI-compatible endpoint.

    Uses the exact call shape requested in the project spec (base_url, seed=).
    """

    def __init__(self, *, api_key: str | None = None, model: str = NEBIUS_MODEL):
        key = api_key or os.environ.get("NEBIUS_API_KEY")
        if not key:
            raise RuntimeError("NEBIUS_API_KEY not set")
        self._client = openai.AsyncOpenAI(base_url=NEBIUS_BASE_URL, api_key=key)
        self._model = model

    @retry(
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    async def complete(
        self,
        *,
        system: str,
        user: str,
        seed: int,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> NebiusResponse:
        resp = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": user}]},
            ],
            seed=seed,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        usage: Any = resp.usage
        return NebiusResponse(
            text=(choice.message.content or "").strip(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
        )

    async def close(self) -> None:
        await self._client.close()
