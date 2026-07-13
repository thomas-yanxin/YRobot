"""Thin streaming client over any OpenAI-compatible endpoint.

Used for both the local text model (``mlx_lm.server``) and the cloud vision model
(ModelScope). Yields text deltas; tolerates providers that stream ``reasoning_content``
before ``content`` (e.g. Qwen thinking models) by ignoring the reasoning stream.
"""
from __future__ import annotations

import logging
from typing import Iterator, List

log = logging.getLogger("live_chat.llm.client")


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self._client = None
        self._api_key = api_key or "not-needed"

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI  # lazy

            self._client = OpenAI(base_url=self.base_url, api_key=self._api_key)
        return self._client

    def stream(
        self,
        messages: List[dict],
        *,
        temperature: float = 0.6,
        max_tokens: int = 300,
        stop_check=None,
    ) -> Iterator[str]:
        """Yield content deltas. ``stop_check()`` truthy aborts the stream (barge-in)."""
        client = self._ensure()
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as e:
            log.warning("LLM request failed (%s): %s", self.model, e)
            return
        for chunk in resp:
            if stop_check is not None and stop_check():
                try:
                    resp.close()
                except Exception:
                    pass
                return
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content
