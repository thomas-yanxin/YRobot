#!/usr/bin/env python3
"""Minimal smoke test for the MiniCPM-o video Realtime API."""

from __future__ import annotations

import argparse
import base64
import json
import math
import ssl
import sys
import time
from pathlib import Path
from typing import Any

from websockets.sync.client import connect

DEFAULT_URL = "wss://10.0.16.184:8006/v1/realtime?mode=video"


def printable(event: dict[str, Any]) -> dict[str, Any]:
    """Hide large base64 fields while keeping the wire event readable."""

    shown = dict(event)
    model_input = shown.get("input")
    if isinstance(model_input, dict) and isinstance(model_input.get("audio"), str):
        shown["input"] = dict(model_input)
        shown["input"]["audio"] = f"<base64: {len(model_input['audio'])} chars>"
        frames = model_input.get("video_frames")
        if isinstance(frames, list):
            shown["input"]["video_frames"] = ["<base64 JPEG>" for _ in frames]
    if isinstance(shown.get("audio"), str):
        shown["audio"] = f"<base64: {len(shown['audio'])} chars>"
    return shown


def receive(websocket: Any, deadline: float) -> dict[str, Any]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("等待服务端事件超时")
    raw = websocket.recv(timeout=remaining)
    event = json.loads(raw)
    if not isinstance(event, dict) or not isinstance(event.get("type"), str):
        raise RuntimeError(f"服务端返回了非法事件: {event!r}")
    print("<-", json.dumps(printable(event), ensure_ascii=False))
    if event["type"] == "error":
        raise RuntimeError(json.dumps(event, ensure_ascii=False))
    return event


def wait_for(websocket: Any, expected: str, deadline: float) -> dict[str, Any]:
    while True:
        event = receive(websocket, deadline)
        event_type = event["type"]
        if event_type == expected:
            return event
        if expected == "session.queue_done" and event_type in {
            "session.queued",
            "session.queue_update",
        }:
            continue
        if expected == "response.output.delta" and event_type == "response.done":
            continue
        if event_type == "session.closed":
            raise RuntimeError(f"会话提前关闭: {event.get('reason', 'unknown')}")


def send(websocket: Any, event: dict[str, Any]) -> None:
    print("->", json.dumps(printable(event), ensure_ascii=False))
    websocket.send(json.dumps(event, ensure_ascii=False, separators=(",", ":")))


def smoke_test(url: str, timeout: float, tls_no_verify: bool, image: Path | None = None) -> None:
    deadline = time.monotonic() + timeout
    ssl_context = None
    if url.startswith("wss://") and tls_no_verify:
        ssl_context = ssl._create_unverified_context()

    started = time.monotonic()
    with connect(
        url,
        ssl=ssl_context,
        open_timeout=min(10.0, timeout),
        close_timeout=3.0,
        max_size=128 * 1024 * 1024,
        compression=None,
    ) as websocket:
        wait_for(websocket, "session.queue_done", deadline)

        send(
            websocket,
            {
                "type": "session.init",
                "payload": {
                    "system_prompt": "You are a helpful assistant.",
                },
            },
        )
        created = wait_for(websocket, "session.created", deadline)

        # 500 ms、16 kHz、单声道 float32 PCM；全零字节就是 float32 的 0.0。
        silence = base64.b64encode(bytes(8_000 * 4)).decode("ascii")
        model_input: dict[str, Any] = {
            "audio": silence,
            "force_listen": True,
            "max_slice_nums": 1,
        }
        if image is not None:
            jpeg = image.read_bytes()
            if not (jpeg.startswith(b"\xff\xd8") and jpeg.endswith(b"\xff\xd9")):
                raise ValueError(f"不是有效的 JPEG 文件: {image}")
            model_input["video_frames"] = [base64.b64encode(jpeg).decode("ascii")]
        send(websocket, {"type": "input.append", "input": model_input})

        while True:
            event = wait_for(websocket, "response.output.delta", deadline)
            if event.get("kind") == "listen":
                break

        send(websocket, {"type": "session.close", "reason": "smoke_test_complete"})
        print(
            "PASS:",
            f"session_id={created.get('session_id', '-')}",
            f"elapsed_ms={(time.monotonic() - started) * 1000:.1f}",
        )


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是正数")
    return number


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--timeout", type=positive_float, default=120.0)
    parser.add_argument("--image", type=Path, help="可选：附带一张 JPEG 以检查视觉输入链路")
    parser.add_argument(
        "--tls-no-verify",
        action="store_true",
        help="仅用于自签名 wss:// 测试服务",
    )
    args = parser.parse_args()
    try:
        smoke_test(args.url, args.timeout, args.tls_no_verify, args.image)
    except KeyboardInterrupt:
        print("FAIL: 用户中断", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
