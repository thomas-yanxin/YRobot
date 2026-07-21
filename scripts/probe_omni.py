#!/usr/bin/env python3
"""Verify a raw llama-omni-server with one deterministic full-duplex time slice."""

from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np
import websockets

from yrobot.config import Config, normalize_backend_url
from yrobot.omni import build_input_append, build_session_init


async def probe(url: str, tls_verify: bool) -> None:
    config = Config(
        omni_url=normalize_backend_url(url),
        tls_verify=tls_verify,
        send_video=False,
        system_prompt="你是 Reachy Mini。保持安静，继续倾听。",
    )
    async with websockets.connect(
        config.omni_url,
        ssl=config.ssl_context(),
        open_timeout=10,
        close_timeout=3,
        max_size=config.max_message_size,
    ) as websocket:
        await websocket.send(json.dumps(build_session_init(config.system_prompt)))
        created = json.loads(await asyncio.wait_for(websocket.recv(), 120))
        print(
            "init",
            created.get("type"),
            created.get("mode"),
            bool(created.get("session_id")),
            flush=True,
        )
        request = build_input_append(np.zeros(16_000, dtype=np.float32), None)
        request["input"]["force_listen"] = True
        await websocket.send(json.dumps(request))
        for _ in range(12):
            event = json.loads(await asyncio.wait_for(websocket.recv(), 120))
            marker = event.get("kind") or event.get("reason") or ""
            print("event", event.get("type"), marker, flush=True)
            if marker == "listen":
                break
            if event.get("type") in {"session.closed", "error"}:
                raise RuntimeError(f"server rejected the probe: {event}")
        else:
            raise RuntimeError("server did not acknowledge the forced-listen time slice")

        # A force-listen acknowledgement must not wedge the session. Verify that
        # the next ordinary microphone slice is still consumed normally.
        await websocket.send(
            json.dumps(build_input_append(np.zeros(16_000, dtype=np.float32), None))
        )
        for _ in range(12):
            event = json.loads(await asyncio.wait_for(websocket.recv(), 120))
            marker = event.get("kind") or event.get("reason") or ""
            print("post-force", event.get("type"), marker, flush=True)
            if marker == "listen":
                return
            if event.get("type") in {"session.closed", "error"}:
                raise RuntimeError(f"server rejected post-force input: {event}")
        raise RuntimeError("server stopped consuming input after force_listen")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="wss://10.0.16.187:28099/backend")
    parser.add_argument("--tls-verify", action="store_true")
    args = parser.parse_args()
    asyncio.run(probe(args.url, args.tls_verify))


if __name__ == "__main__":
    main()
