import asyncio
import json
import threading
import time

import numpy as np
import websockets

from yrobot.config import Config
from yrobot.omni import OmniClient, encode_pcm


class FakeRobot:
    def __init__(self) -> None:
        self.sent = False
        self.flushed = False
        self.played: list[np.ndarray] = []
        self.states: list[str] = []

    def next_audio_chunk(self, timeout: float) -> np.ndarray | None:
        if self.flushed and not self.sent:
            self.sent = True
            return np.zeros(16_000, dtype=np.float32)
        time.sleep(min(timeout, 0.01))
        return None

    def flush_audio_input(self) -> None:
        self.flushed = True

    def get_frame_jpeg(self) -> bytes:
        return b"jpeg"

    def play_omni_audio(self, samples: np.ndarray) -> None:
        self.played.append(samples)

    def set_conversation_state(self, state: str) -> None:
        self.states.append(state)


def test_full_duplex_session_end_to_end() -> None:
    asyncio.run(_session_scenario())


async def _session_scenario() -> None:
    stop_event = threading.Event()
    observed: list[dict[str, object]] = []

    async def handler(websocket: object) -> None:
        observed.append(json.loads(await websocket.recv()))
        await websocket.send(
            json.dumps({"type": "session.created", "session_id": "test", "mode": "full_duplex"})
        )
        observed.append(json.loads(await websocket.recv()))
        await websocket.send(
            json.dumps(
                {
                    "type": "response.output.delta",
                    "kind": "text",
                    "response_id": "r1",
                    "text": "你好",
                }
            )
        )
        await websocket.send(
            json.dumps(
                {
                    "type": "response.output.delta",
                    "kind": "audio",
                    "response_id": "r1",
                    "audio": encode_pcm(np.linspace(-0.2, 0.2, 240, dtype=np.float32)),
                }
            )
        )
        await websocket.send(
            json.dumps(
                {
                    "type": "response.done",
                    "response_id": "r1",
                    "text": "你好",
                    "reason": "turn_end",
                }
            )
        )
        await asyncio.sleep(0.1)
        stop_event.set()

    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        config = Config(
            omni_url=f"ws://127.0.0.1:{port}/backend",
            tls_verify=True,
            send_video=True,
            system_prompt="test",
        )
        robot = FakeRobot()
        await OmniClient(config).run_session(robot, stop_event)

    assert observed[0]["type"] == "session.init"
    assert observed[1]["type"] == "input.append"
    assert len(robot.played) == 1
    np.testing.assert_allclose(robot.played[0], np.linspace(-0.2, 0.2, 240))
    assert robot.states[0] == "listening"
    assert "speaking" in robot.states
