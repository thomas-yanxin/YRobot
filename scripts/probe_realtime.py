#!/usr/bin/env python3
"""Smoke-test the public MiniCPM-o video Realtime Gateway lifecycle."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import websockets

from yrobot.config import DEFAULT_SYSTEM_PROMPT, Config, normalize_realtime_url
from yrobot.realtime import (
    RealtimeProtocolError,
    build_input_append,
    build_session_init,
)

DEFAULT_PROBE_URL = "wss://10.0.16.184:8006/v1/realtime?mode=video"


def _parse_event(raw: str | bytes) -> dict[str, Any]:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    event = json.loads(raw)
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise RealtimeProtocolError("Realtime event must be an object with a string type")
    return event


def _raise_if_terminal(event: Mapping[str, Any], stage: str) -> None:
    event_type = event.get("type")
    if event_type == "error":
        raise RealtimeProtocolError(json.dumps(event, ensure_ascii=False))
    if event_type == "session.closed":
        reason = event.get("reason") or "no reason"
        raise ConnectionError(f"session closed during {stage}: {reason}")


async def _recv_event(websocket: Any, deadline: float) -> dict[str, Any]:
    remaining = deadline - asyncio.get_running_loop().time()
    if remaining <= 0.0:
        raise TimeoutError("Realtime probe timed out")
    raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
    return _parse_event(raw)


async def _wait_for_queue_done(
    websocket: Any,
    deadline: float,
) -> tuple[int, dict[str, Any]]:
    updates = 0
    while True:
        event = await _recv_event(websocket, deadline)
        event_type = event["type"]
        _raise_if_terminal(event, "queue wait")
        if event_type in {"session.queued", "session.queue_update"}:
            updates += 1
            continue
        if event_type == "session.queue_done":
            return updates, event
        if event_type == "session.created":
            raise RealtimeProtocolError("session.created arrived before session.queue_done")
        raise RealtimeProtocolError(f"unexpected event before session.queue_done: {event_type}")


async def _wait_for_created(websocket: Any, deadline: float) -> dict[str, Any]:
    while True:
        event = await _recv_event(websocket, deadline)
        event_type = event["type"]
        _raise_if_terminal(event, "session initialization")
        if event_type == "session.created":
            if not event.get("session_id"):
                raise RealtimeProtocolError("session.created omitted session_id")
            return event
        raise RealtimeProtocolError(f"unexpected event after session.init: {event_type}")


async def _wait_for_listen(websocket: Any, deadline: float) -> dict[str, Any]:
    while True:
        event = await _recv_event(websocket, deadline)
        event_type = event["type"]
        _raise_if_terminal(event, "forced-listen response")
        if event_type == "response.output.delta":
            if event.get("kind") == "listen":
                return event
            # Text/audio can race a force-listen decision; the smoke condition
            # is the explicit full-duplex listen boundary.
            continue
        if event_type == "response.done":
            # response.done is not the video-duplex turn boundary.
            continue


async def probe(url: str, *, tls_verify: bool, timeout: float) -> None:
    """Run one queued session and one deterministic force-listen audio unit."""

    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ValueError("timeout must be a positive finite number")
    config = Config(
        realtime_url=normalize_realtime_url(url),
        tls_verify=tls_verify,
        send_video=False,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        handshake_timeout=timeout,
    )

    started_at = time.perf_counter()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    websocket: Any | None = None
    try:
        websocket = await websockets.connect(
            config.realtime_url,
            ssl=config.ssl_context(),
            open_timeout=min(10.0, timeout),
            close_timeout=min(3.0, timeout),
            max_size=config.max_message_size,
            max_queue=16,
            compression=None,
            ping_interval=20,
            ping_timeout=20,
        )
        connected_at = time.perf_counter()

        # The Gateway owns scarce model slots. Sending init before queue_done is
        # a protocol violation even when the queue appears empty.
        queue_updates, queue_done = await _wait_for_queue_done(websocket, deadline)
        queue_done_at = time.perf_counter()
        init = build_session_init(
            config.system_prompt,
            length_penalty=config.length_penalty,
            force_listen_count=config.force_listen_count,
            enable_tts=False,
        )
        await websocket.send(json.dumps(init, separators=(",", ":")))
        init_sent_at = time.perf_counter()

        created = await _wait_for_created(websocket, deadline)
        created_at = time.perf_counter()

        silence = np.zeros(config.audio_unit_samples, dtype=np.float32)
        request = build_input_append(
            silence,
            None,
            force_listen=True,
        )
        await websocket.send(json.dumps(request, separators=(",", ":")))
        input_sent_at = time.perf_counter()

        listen = await _wait_for_listen(websocket, deadline)
        listen_at = time.perf_counter()
        raw_metrics = listen.get("metrics")
        metrics = dict(raw_metrics) if isinstance(raw_metrics, Mapping) else {}

        print("session", str(created["session_id"]), flush=True)
        print(
            "queue",
            json.dumps(
                {
                    "updates": queue_updates,
                    "event": queue_done.get("type"),
                    "wait_ms": round((queue_done_at - connected_at) * 1_000, 1),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )
        print(
            "init",
            json.dumps(
                {
                    "sent_after_queue_done": init_sent_at >= queue_done_at,
                    "created_ms": round((created_at - init_sent_at) * 1_000, 1),
                    "force_listen": request["input"]["force_listen"],
                    "audio_samples": silence.size,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )
        print(
            "metrics",
            json.dumps(metrics, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
        print(
            "wall_clock_ms",
            metrics.get("wall_clock_ms"),
            "client_total_ms",
            round((listen_at - started_at) * 1_000, 1),
            "input_to_listen_ms",
            round((listen_at - input_sent_at) * 1_000, 1),
            flush=True,
        )
    finally:
        if websocket is not None:
            with contextlib.suppress(Exception):
                await websocket.send(
                    json.dumps(
                        {"type": "session.close", "reason": "smoke_probe_complete"},
                        separators=(",", ":"),
                    )
                )
                # The worker only becomes reusable after its backend close
                # finishes. session.closed starts the final cleanup; transport
                # closure confirms that the Gateway has released the worker.
                async with asyncio.timeout(5.0):
                    while True:
                        event = _parse_event(await websocket.recv())
                        if event["type"] == "session.closed":
                            break
                    # The immutable Gateway releases the worker before it
                    # closes the client transport. Waiting here avoids racing
                    # the next probe against that final cleanup.
                    await websocket.wait_closed()
            with contextlib.suppress(Exception):
                await websocket.close()


def _positive_timeout(value: str) -> float:
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return timeout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_PROBE_URL)
    tls = parser.add_mutually_exclusive_group()
    tls.add_argument(
        "--tls-verify",
        dest="tls_verify",
        action="store_true",
        help="verify the wss:// certificate (recommended after installing a trusted cert)",
    )
    tls.add_argument(
        "--tls-no-verify",
        dest="tls_verify",
        action="store_false",
        help="accept the configured LAN Gateway's self-signed certificate (default)",
    )
    parser.set_defaults(tls_verify=False)
    parser.add_argument(
        "--timeout",
        type=_positive_timeout,
        default=120.0,
        help="overall probe deadline in seconds (default: 120)",
    )
    args = parser.parse_args()
    asyncio.run(
        probe(
            args.url,
            tls_verify=args.tls_verify,
            timeout=args.timeout,
        )
    )


if __name__ == "__main__":
    main()
