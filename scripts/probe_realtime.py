#!/usr/bin/env python3
"""Probe the documented MiniCPM-o video Realtime lifecycle."""

from __future__ import annotations

import argparse
import ssl
import time
from pathlib import Path

import numpy as np
from websockets.sync.client import connect

from yrobot.config import OFFICIAL_REALTIME_URL
from yrobot.protocol import (
    ProtocolState,
    QueueDone,
    QueueStatus,
    ResponseDelta,
    ServerError,
    SessionClosed,
    SessionCreated,
    input_append,
    parse_server_event,
    serialize_client_event,
    session_close,
    session_init,
    transition_client,
    transition_server,
    validate_video_url,
)


def probe(
    url: str,
    *,
    seconds: int,
    image: Path | None,
    tls_verify: bool,
) -> None:
    validate_video_url(url)
    frame = image.read_bytes() if image else None
    if frame is not None and not (frame.startswith(b"\xff\xd8") and frame.endswith(b"\xff\xd9")):
        raise ValueError(f"not a JPEG: {image}")

    context = None
    if url.startswith("wss://"):
        context = ssl.create_default_context()
        if not tls_verify:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

    state = ProtocolState()
    started = time.monotonic()
    with connect(
        url,
        ssl=context,
        open_timeout=10,
        close_timeout=3,
        max_size=16 * 1024 * 1024,
        compression=None,
    ) as websocket:
        while True:
            event = parse_server_event(websocket.recv(timeout=120))
            state = transition_server(state, event)
            if isinstance(event, QueueStatus):
                print(f"queue position={event.position}")
            elif isinstance(event, QueueDone):
                break
            elif isinstance(event, ServerError):
                raise RuntimeError(event.error)

        init = session_init("Reply briefly and naturally.", length_penalty=1.1)
        websocket.send(serialize_client_event(init))
        state = transition_client(state, init)
        created: SessionCreated | None = None
        while created is None:
            event = parse_server_event(websocket.recv(timeout=120))
            state = transition_server(state, event)
            if isinstance(event, SessionCreated):
                created = event
            elif isinstance(event, ServerError):
                raise RuntimeError(event.error)

        silence = np.zeros(16_000, dtype="<f4").tobytes()
        next_send = time.monotonic()
        for index in range(seconds):
            if next_send > time.monotonic():
                time.sleep(next_send - time.monotonic())
            event = input_append(
                silence,
                video_frames=(frame,) if frame else (),
                force_listen=index == 0,
                max_slice_nums=1,
            )
            send_started = time.monotonic()
            websocket.send(serialize_client_event(event))
            state = transition_client(state, event)
            completed = time.monotonic()
            next_send = send_started + 1.0
            if next_send <= completed:
                next_send = completed + 1.0

        listen_seen = False
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not listen_seen:
            event = parse_server_event(
                websocket.recv(timeout=max(0.1, deadline - time.monotonic()))
            )
            state = transition_server(state, event)
            if isinstance(event, ResponseDelta) and event.kind == "listen":
                listen_seen = True
            elif isinstance(event, ServerError):
                raise RuntimeError(event.error)

        close = session_close("probe_complete")
        websocket.send(serialize_client_event(close))
        state = transition_client(state, close)
        while True:
            event = parse_server_event(websocket.recv(timeout=5))
            if isinstance(event, SessionClosed):
                state = transition_server(state, event)
                break

    print(
        f"PASS session={created.session_id} listen={listen_seen} "
        f"elapsed={time.monotonic() - started:.2f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=OFFICIAL_REALTIME_URL)
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--tls-no-verify", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 30:
        parser.error("--seconds must be 1..30")
    probe(
        args.url,
        seconds=args.seconds,
        image=args.image,
        tls_verify=not args.tls_no_verify,
    )


if __name__ == "__main__":
    main()
