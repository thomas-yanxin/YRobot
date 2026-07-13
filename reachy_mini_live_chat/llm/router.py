"""Conversation engine: routes text vs vision, streams clauses + emotion.

* Local text model for normal turns (low latency).
* Cloud/local **vision** model only when a gated keyframe is attached (token saving).
* Parses the leading ``<emo>`` tag, strips it from speech, and emits it separately.
* Enforces the **single-keyframe** context rule: at most one image is ever kept in history.
* Trims history to a few turns to keep prompts (and TTFT) small.
"""
from __future__ import annotations

import logging
import re
from typing import Iterator, List, Optional

from ..config import Config
from ..text_utils import ClauseAccumulator, clean_spoken
from .client import OpenAICompatClient
from .prompts import ALLOWED_EMOTIONS, SYSTEM_PROMPT

log = logging.getLogger("live_chat.llm")

MAX_TURNS = 6  # user+assistant messages kept (besides system)


class LlmEngine:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._text = OpenAICompatClient(cfg.llm_base_url, cfg.llm_api_key, cfg.llm_model, cfg.no_think)
        self._vision = OpenAICompatClient(cfg.vision_base_url, cfg.vision_api_key, cfg.vision_model, cfg.no_think)
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
                clause = clean_spoken(clause)
                if clause:
                    spoken.append(clause)
                    yield {"type": "clause", "text": clause}

        if not emo_done and prefix:
            # stream ended before we decided; treat whole prefix as text
            for clause in acc.push(prefix):
                clause = clean_spoken(clause)
                if clause:
                    spoken.append(clause)
                    yield {"type": "clause", "text": clause}
        tail = clean_spoken(acc.flush() or "")
        if tail:
            spoken.append(tail)
            yield {"type": "clause", "text": tail}

        full = " ".join(spoken).strip()
        self._record_assistant(full)
        yield {"type": "final", "text": full}


_NAME_RE = re.compile(r"[A-Za-z0-9_]+")


def _scan_emotion(prefix: str):
    """Decide whether ``prefix`` begins with a ``<emo>NAME`` tag (closing tag optional).

    Small models often drop the ``</emo>`` and/or bolt on markdown (``<emo>yes1**...``), so we
    accept an unterminated tag: read the NAME word after ``<emo>`` and treat the first
    non-word char as the end. Returns (name|None, remainder_text, decided); ``decided`` is False
    while more characters are still needed to tell.
    """
    s = prefix.lstrip()
    if not s:
        return None, "", False
    if not s.startswith("<"):
        return None, prefix, True                      # definitely no tag
    if not s.startswith("<emo>"):
        # still possibly building "<", "<e", "<em", "<emo"
        if "<emo>".startswith(s):
            return None, "", False
        return None, prefix, True                      # a '<' but not our tag (e.g. "<b>")
    rest = s[len("<emo>"):]
    m = _NAME_RE.match(rest)
    name = m.group(0) if m else ""
    after = rest[len(name):]
    if after == "":
        # buffer ends inside/at the name — it may still grow (e.g. "ye" -> "yes1")
        return (None, prefix, True) if len(s) > 40 else (None, "", False)
    # a possibly-splitting closing tag: wait for the rest of "</emo>"
    if after != "</emo>" and "</emo>".startswith(after) and len(s) <= 60:
        return None, "", False
    remainder = after[len("</emo>"):] if after.startswith("</emo>") else after
    if name not in ALLOWED_EMOTIONS:
        name = None
    return name, remainder, True


def _stub_stream(user_text: str, lang: str) -> Iterator[str]:
    """Deterministic canned reply so the pipeline is demoable without any model."""
    if lang == "zh":
        yield from ["<emo>cheerful1</emo>", "好的，", "我听到你说“", user_text[:20], "”。", "这是模拟回复。"]
    else:
        yield from ["<emo>cheerful1</emo>", "Got it — ", "I heard you say “", user_text[:30], "”. ", "This is a simulated reply."]
