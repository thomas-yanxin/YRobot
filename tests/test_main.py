from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np

from yrobot.config import Settings
from yrobot.main import Yrobot, cli
from yrobot.runtime import YRobotRuntime


class FakeAudioBackend:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.max_buffers = 0

    def set_max_output_buffers(self, value: int) -> None:
        self.max_buffers = value
        self.events.append("audio.buffers")

    def clear_player(self) -> None:
        self.events.append("audio.clear")


class FakeMedia:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.audio = FakeAudioBackend(events)

    def start_playing(self) -> None:
        self.events.append("media.play")

    def start_recording(self) -> None:
        self.events.append("media.record")

    def stop_playing(self) -> None:
        self.events.append("media.stop_play")

    def stop_recording(self) -> None:
        self.events.append("media.stop_record")

    def get_audio_sample(self) -> None:
        return None

    def push_audio_sample(self, _samples: np.ndarray) -> None: ...

    def get_frame(self) -> None:
        return None

    def get_DoA(self) -> None:  # noqa: N802
        return None


class FakeMini:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.media = FakeMedia(events)

    def enable_motors(self) -> None:
        self.events.append("motors.enable")

    def enable_wobbling(self) -> None:
        self.events.append("wobble.enable")

    def disable_wobbling(self) -> None:
        self.events.append("wobble.disable")

    def get_current_head_pose(self) -> np.ndarray:
        return np.eye(4)

    def get_present_antenna_joint_positions(self) -> list[float]:
        return [0.0, 0.0]

    def set_target(self, **_kwargs: Any) -> None:
        self.events.append("motion.target")


def test_runtime_owns_media_once_and_stops_in_dependency_order(
    monkeypatch: Any,
) -> None:
    events: list[str] = []

    class Client:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.metrics = {"sessions": 1}

        async def run(self, stop_event: threading.Event) -> None:
            events.append("client.run")
            stop_event.set()
            await asyncio.sleep(0)

        def submit_audio(self, *_args: Any, **_kwargs: Any) -> None: ...

    monkeypatch.setattr("yrobot.runtime.RealtimeClient", Client)
    runtime = YRobotRuntime(
        FakeMini(events),
        Settings(realtime_url="ws://brain.local/v1/realtime?mode=video"),
        threading.Event(),
    )
    runtime.run()

    assert events.index("motors.enable") < events.index("media.play")
    assert events.index("media.play") < events.index("media.record")
    assert events.index("media.record") < events.index("client.run")
    assert events.index("client.run") < events.index("media.stop_play")
    assert events.index("wobble.disable") < events.index("media.stop_play")
    assert events[-2:] == ["media.stop_play", "media.stop_record"]
    assert events.count("media.play") == 1
    assert events.count("media.record") == 1
    assert "audio.clear" in events


def test_reachy_app_requests_the_wireless_local_media_backend() -> None:
    assert Yrobot.request_media_backend == "local"
    assert Yrobot.dont_start_webserver is True
    assert Yrobot.custom_app_url is None


def test_cli_connects_locally_with_automatic_body_yaw(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}
    settings = Settings(realtime_url="ws://brain.local/v1/realtime?mode=video")

    class MiniContext:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        def __enter__(self) -> object:
            return object()

        def __exit__(self, *_args: Any) -> None: ...

    monkeypatch.setattr("yrobot.main.ReachyMini", MiniContext)
    monkeypatch.setattr("yrobot.main.Settings.from_env", lambda: settings)
    monkeypatch.setattr(
        "yrobot.main.run_conversation",
        lambda _mini, selected, _stop: seen.update(settings=selected),
    )

    cli(["--url", "ws://other.local/v1/realtime?mode=video"])

    assert seen["connection_mode"] == "localhost_only"
    assert seen["media_backend"] == "local"
    assert seen["automatic_body_yaw"] is True
    assert seen["settings"].realtime_url == "ws://other.local/v1/realtime?mode=video"
