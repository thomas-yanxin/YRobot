"""Remote end-to-end omni brain: WebSocket client for llama.cpp-omni's ``/backend``.

* :mod:`.protocol` — pure, dependency-light codec for the WS wire format (base64
  float32 PCM, ``session.init`` / ``input.append`` builders, event parsing).
* :mod:`.client` — :class:`OmniClient`, an asyncio WebSocket client run on its own
  thread that streams 1 s audio chunks (+ a video frame) up and dispatches text /
  audio / listen / done events to a sink.
* :mod:`.video` — :class:`VideoGrabber`, camera frame → downscaled base64 JPEG.
* :mod:`.fake_server` — a tiny local omni server for ``--stub`` / tests.
"""
from .protocol import (
    OmniEvent,
    b64_to_pcm_f32,
    build_input_append,
    build_session_init,
    parse_event,
    pcm_f32_to_b64,
)

__all__ = [
    "OmniEvent",
    "b64_to_pcm_f32",
    "build_input_append",
    "build_session_init",
    "parse_event",
    "pcm_f32_to_b64",
]
