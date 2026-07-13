"""Thin streaming client over any OpenAI-compatible endpoint.

Default backend is **MiniCPM-V-4.6 served by llama.cpp** (`llama-server`), used for both
text and vision turns; a cloud VLM works too. Yields *content* deltas only.

MiniCPM-V-4.6 is a reasoning model: with thinking enabled it emits a long
``reasoning_content`` before the answer, which wrecks time-to-first-audio. Run the server
with ``--reasoning off`` (see docs). As defense-in-depth we also (a) ignore any
``reasoning_content`` field and (b) strip ``<think>...</think>`` spans that some templates
leak into ``content`` — so the reasoning trace is never spoken even if the flag is missing.
"""
from __future__ import annotations

import logging
from typing import Iterator, List

log = logging.getLogger("live_chat.llm.client")

_OPEN, _CLOSE = "<think>", "</think>"


class OpenAICompatClient:
    def __init__(self, base_url: str, api_key: str, model: str, disable_thinking: bool = True) -> None:
        self.base_url = base_url
        self.model = model
        self.disable_thinking = disable_thinking
        self._client = None
        self._api_key = api_key or "not-needed"
        self._thinking_kwargs_unsupported = False

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
        max_tokens: int = 400,
        stop_check=None,
    ) -> Iterator[str]:
        """Yield spoken-content deltas. ``stop_check()`` truthy aborts (barge-in)."""
        yield from strip_think(self._raw(messages, temperature, max_tokens, stop_check))

    def _raw(self, messages, temperature, max_tokens, stop_check) -> Iterator[str]:
        client = self._ensure()
        base = dict(model=self.model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens, stream=True)
        # Disable the model's thinking mode at the request level (server-flag-independent).
        use_kwargs = self.disable_thinking and not self._thinking_kwargs_unsupported
        try:
            if use_kwargs:
                resp = client.chat.completions.create(
                    **base, extra_body={"chat_template_kwargs": {"enable_thinking": False}}
                )
            else:
                resp = client.chat.completions.create(**base)
        except Exception as e:
            if use_kwargs:
                # endpoint rejected chat_template_kwargs — remember and retry without it
                log.info("endpoint rejected enable_thinking kwarg (%s); retrying without", e)
                self._thinking_kwargs_unsupported = True
                try:
                    resp = client.chat.completions.create(**base)
                except Exception as e2:
                    log.warning("LLM request failed (%s): %s", self.model, e2)
                    return
            else:
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
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                yield content


def _partial_suffix_len(buf: str, tag: str) -> int:
    """Length of the longest suffix of ``buf`` that is a prefix of ``tag`` (split-tag guard)."""
    for k in range(min(len(buf), len(tag) - 1), 0, -1):
        if buf[-k:] == tag[:k]:
            return k
    return 0


def strip_think(chunks: Iterator[str]) -> Iterator[str]:
    """Drop ``<think>...</think>`` spans from a streamed text iterator (tags may span chunks)."""
    buf = ""
    in_think = False
    for delta in chunks:
        buf += delta
        emit = ""
        progress = True
        while progress:
            progress = False
            if not in_think:
                i = buf.find(_OPEN)
                if i != -1:
                    emit += buf[:i]
                    buf = buf[i + len(_OPEN):]
                    in_think = True
                    progress = True
            else:
                j = buf.find(_CLOSE)
                if j != -1:
                    buf = buf[j + len(_CLOSE):]
                    in_think = False
                    progress = True
        if not in_think:
            keep = _partial_suffix_len(buf, _OPEN)
            emit += buf[: len(buf) - keep]
            buf = buf[len(buf) - keep:]
        else:
            keep = _partial_suffix_len(buf, _CLOSE)
            buf = buf[len(buf) - keep:]
        if emit:
            yield emit
    if buf and not in_think:
        yield buf
