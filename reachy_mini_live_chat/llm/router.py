"""Conversation engine: routes text vs vision, streams clauses + emotion.

* Local text model for normal turns (low latency).
* Cloud/local **vision** model only when a gated keyframe is attached (token saving).
* Parses the leading ``<emo>`` tag, strips it from speech, and emits it separately.
* Enforces the **single-keyframe** context rule: at most one image is ever kept in history.
* Trims history to a few turns to keep prompts (and TTFT) small.
"""
from __future__ import annotations

import logging
from typing import Iterator, List, Optional

from ..config import Config
from ..text_utils import ClauseAccumulator
from .client import OpenAICompatClient
from .prompts import ALLOWED_EMOTIONS, SYSTEM_PROMPT

log = logging.getLogger("live_chat.llm")

MAX_TURNS = 6  # user+assistant messages kept (besides system)


class LlmEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._text = OpenAICompatClient(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model)
        self._vision = OpenAICompatClient(cfg.vision_base_url, cfg.vision_api_key, cfg.vision_model)
        self._history: List[dict] = []

    # -- public -------------------------------------------------------------
    def respond(
        self,
        user_text: str,
        lang: str,
        keyframe_b64: Optional[str] = None,
        stop_check=None,
    ) -> Iterator[dict]:
        """Yield events: {'type':'emotion','name'} | {'type':'clause','text'} | {'type':'final','text'}."""
        user_msg = self._make_user_msg(user_text, keyframe_b64)
        self._append(user_msg)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self._history

        if self.cfg.stub:
            stream = _stub_stream(user_text, lang)
        elif keyframe_b64 is not None:
            # Vision turn: attach the gated keyframe. By default this is the same local
            # MiniCPM-V server; can be pointed at a cloud VLM via VISION_* config.
            log.info("vision turn: attaching keyframe -> %s", self.cfg.vision_model)
            stream = self._vision.stream(messages, stop_check=stop_check)
        else:
            stream = self._text.stream(messages, stop_check=stop_check)

        yield from self._parse_stream(stream)

    # -- history ------------------------------------------------------------
    def _make_user_msg(self, text: str, keyframe_b64: Optional[str]) -> dict:
        if keyframe_b64 is None:
            return {"role": "user", "content": text}
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{keyframe_b64}"}},
            ],
        }

    def _append(self, msg: dict) -> None:
        # single-keyframe rule: strip images from older messages before adding a new turn
        for m in self._history:
            if isinstance(m.get("content"), list):
                m["content"] = "".join(
                    part.get("text", "") for part in m["content"] if part.get("type") == "text"
                ) or "[image]"
        self._history.append(msg)
        if len(self._history) > MAX_TURNS:
            self._history = self._history[-MAX_TURNS:]

    def _record_assistant(self, text: str) -> None:
        if text:
            self._history.append({"role": "assistant", "content": text})

    # -- streaming parse ----------------------------------------------------
    def _parse_stream(self, stream: Iterator[str]) -> Iterator[dict]:
        acc = ClauseAccumulator()
        spoken: List[str] = []
        emo_done = False
        prefix = ""

        for delta in stream:
            if not emo_done:
                prefix += delta
                name, remainder, decided = _scan_emotion(prefix)
                if not decided:
                    continue
                emo_done = True
                if name:
                    yield {"type": "emotion", "name": name}
                delta = remainder  # feed leftover text into the clause splitter
                prefix = ""
            for clause in acc.push(delta):
                spoken.append(clause)
                yield {"type": "clause", "text": clause}

        if not emo_done and prefix:
            # stream ended before we decided; treat whole prefix as text
            for clause in acc.push(prefix):
                spoken.append(clause)
                yield {"type": "clause", "text": clause}
        tail = acc.flush()
        if tail:
            spoken.append(tail)
            yield {"type": "clause", "text": tail}

        full = " ".join(spoken).strip()
        self._record_assistant(full)
        yield {"type": "final", "text": full}


def _scan_emotion(prefix: str):
    """Decide whether ``prefix`` begins with a <emo>..</emo> tag.

    Returns (name|None, remainder_text, decided). ``decided`` is False while we still
    need more characters to tell.
    """
    s = prefix.lstrip()
    if not s:
        return None, "", False
    if not s.startswith("<"):
        return None, prefix, True  # definitely no tag
    if not s.startswith("<emo>"):
        # could still be building up "<em"...
        if len("<emo>") <= len(s):
            return None, prefix, True  # it's a '<' but not our tag
        return None, "", False
    end = s.find("</emo>")
    if end == -1:
        if len(s) > 40:  # runaway; give up
            return None, prefix, True
        return None, "", False
    name = s[len("<emo>"):end].strip()
    remainder = s[end + len("</emo>"):]
    if name not in ALLOWED_EMOTIONS:
        name = None
    return name, remainder, True


def _stub_stream(user_text: str, lang: str) -> Iterator[str]:
    """Deterministic canned reply so the pipeline is demoable without any model."""
    if lang == "zh":
        yield from ["<emo>cheerful1</emo>", "好的，", "我听到你说“", user_text[:20], "”。", "这是模拟回复。"]
    else:
        yield from ["<emo>cheerful1</emo>", "Got it — ", "I heard you say “", user_text[:30], "”. ", "This is a simulated reply."]
