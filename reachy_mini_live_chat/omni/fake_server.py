"""A tiny in-process omni WebSocket server for ``--stub`` and tests.

It speaks just enough of the real ``/backend`` protocol to exercise the *real*
:class:`~reachy_mini_live_chat.omni.client.OmniClient` end-to-end with no GPU server
and no hardware: it accepts ``session.init`` → ``session.created``, then for each
``input.append`` it mostly replies with a ``listen`` delta and, every few chunks,
"speaks" — streaming ``text`` deltas + a short beep as an ``audio`` delta + a
``response.done``. That drives the transcript, the speaker, and the motion moods.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Callable, Tuple

import numpy as np

from . import protocol

log = logging.getLogger("live_chat.omni.fake")

_CANNED = [
    "你好，我在听，随时可以聊。",
    "嗯，我明白了。",
    "我能看到你，我们继续吧。",
    "Sure — I'm right here with you.",
]


def _pieces(text: str, n: int = 3):
    return [text[i:i + n] for i in range(0, len(text), n)]


def _beep(sr: int, seconds: float, freq: float = 330.0) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / sr
    return (0.12 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


async def _handle(ws, out_sr: int, speak_every: int) -> None:
    try:
        raw = await ws.recv()
        init = json.loads(raw)
    except Exception:
        return
    mode = (init.get("payload") or {}).get("mode", "full_duplex")
    await ws.send(json.dumps({"type": "session.created", "session_id": "fake", "mode": mode}))

    count = 0
    resp = 0
    async for raw in ws:
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if msg.get("type") != "input.append":
            continue
        count += 1
        if count % max(1, speak_every) != 0:
            await ws.send(json.dumps({
                "type": "response.output.delta", "kind": "listen", "session_id": "fake",
            }))
            continue
        resp += 1
        rid = f"fake_resp_{resp}"
        text = _CANNED[resp % len(_CANNED)]
        for piece in _pieces(text):
            await ws.send(json.dumps({
                "type": "response.output.delta", "kind": "text",
                "session_id": "fake", "response_id": rid, "text": piece,
            }))
            await asyncio.sleep(0.01)
        beep = _beep(out_sr, 0.4)
        await ws.send(json.dumps({
            "type": "response.output.delta", "kind": "audio",
            "session_id": "fake", "response_id": rid, "audio": protocol.pcm_f32_to_b64(beep),
        }))
        await ws.send(json.dumps({
            "type": "response.done", "session_id": "fake", "response_id": rid,
            "text": text, "reason": "turn_end", "audio": None,
        }))


def serve_in_thread(
    host: str = "127.0.0.1",
    port: int = 0,
    out_sr: int = 24000,
    speak_every: int = 3,
) -> Tuple[threading.Thread, Callable[[], None], int]:
    """Start the fake server on its own thread. Returns (thread, stop_fn, actual_port).

    ``port=0`` lets the OS pick a free port (read back from the bound socket).
    """
    ready = threading.Event()
    holder: dict = {}

    def run() -> None:
        try:
            import websockets
        except Exception as e:  # pragma: no cover
            log.error("fake omni server needs `websockets` (%s)", e)
            ready.set()
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def main() -> None:
            async def handler(ws, *_a):  # tolerate old/new websockets handler signatures
                await _handle(ws, out_sr, speak_every)

            server = await websockets.serve(handler, host, port, max_size=None)
            holder["loop"] = loop
            holder["server"] = server
            holder["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            await asyncio.Future()  # run until cancelled

        try:
            loop.run_until_complete(main())
        except Exception:
            pass
        finally:
            try:
                loop.close()
            except Exception:
                pass

    thread = threading.Thread(target=run, name="fake-omni", daemon=True)
    thread.start()
    ready.wait(timeout=5)

    def stop() -> None:
        loop = holder.get("loop")
        server = holder.get("server")
        if loop and server:
            loop.call_soon_threadsafe(server.close)

    return thread, stop, int(holder.get("port", port))
