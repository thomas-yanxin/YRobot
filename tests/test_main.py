from __future__ import annotations

import asyncio
import logging
import threading
from logging.handlers import QueueHandler
from typing import Any

import numpy as np

from yrobot.config import Settings
from yrobot.main import Yrobot, cli, configure_logging, shutdown_logging
from yrobot.perception import LatestFrame
from yrobot.runtime import YRobotRuntime, _VisionUplink


class FakeAudioBackend:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.max_buffers = 0

    def set_max_output_buffers(self, value: int) -> None:
        self.max_buffers = value
        self.events.append("audio.buffers")

    def clear_player(self) -> None:
        self.events.append("audio.clear")

    def apply_audio_config(
        self,
        _config: object,
        *,
        verify: bool,
        write_settle_seconds: float,
    ) -> bool:
        assert verify is True
        assert write_settle_seconds == 0.1
        self.events.append("audio.config")
        return True


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
        self.client = type("Client", (), {"host": "127.0.0.1", "port": 8000})()

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

    class DoA:
        def __init__(self, source: str, *_args: Any, **_kwargs: Any) -> None:
            assert source == "http://127.0.0.1:8000/api/state/doa"

        def start(self) -> None:
            events.append("doa.start")

        def stop(self, timeout: float = 2.0) -> bool:
            del timeout
            events.append("doa.stop")
            return True

    monkeypatch.setattr("yrobot.runtime.RealtimeClient", Client)
    monkeypatch.setattr("yrobot.runtime.DoAWorker", DoA)
    runtime = YRobotRuntime(
        FakeMini(events),
        Settings(realtime_url="ws://brain.local/v1/realtime?mode=video"),
        threading.Event(),
    )
    runtime.run()

    assert events.index("motors.enable") < events.index("media.play")
    assert events.index("media.play") < events.index("media.record")
    assert events.index("media.record") < events.index("audio.config")
    assert events.index("audio.config") < events.index("client.run")
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


def test_logging_queue_covers_sdk_and_yrobot_loggers() -> None:
    root = logging.getLogger()
    package = logging.getLogger("yrobot")
    original_root_level = root.level
    original_handlers = tuple(root.handlers)
    original_package_state = (
        package.level,
        tuple(package.handlers),
        package.propagate,
    )

    configure_logging("INFO")
    try:
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], QueueHandler)
        assert logging.getLogger("reachy_mini.media.audio_gstreamer").propagate
        assert logging.getLogger("yrobot.realtime").propagate
    finally:
        shutdown_logging()

    assert tuple(root.handlers) == original_handlers
    assert root.level == original_root_level
    assert (
        package.level,
        tuple(package.handlers),
        package.propagate,
    ) == original_package_state


def test_vision_uplink_sends_only_new_frames_at_model_cadence() -> None:
    now = [1.0]
    latest = LatestFrame()
    uplink = _VisionUplink(latest, 2.0, clock=lambda: now[0])

    latest.publish(b"first", captured_at=now[0])
    assert uplink.next_jpeg() == b"first"
    assert uplink.next_jpeg() is None

    now[0] = 2.0
    latest.publish(b"second", captured_at=now[0])
    assert uplink.next_jpeg() is None

    now[0] = 2.999
    assert uplink.next_jpeg() == b"second"
    assert uplink.next_jpeg() is None

    uplink.reset()
    assert uplink.next_jpeg() == b"second"


def test_runtime_pre_stopped_run_closes_doa_without_starting_hardware(
    monkeypatch: Any,
) -> None:
    events: list[str] = []

    class DoA:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None: ...

        def stop(self, timeout: float = 2.0) -> bool:
            del timeout
            events.append("doa.stop")
            return True

    monkeypatch.setattr("yrobot.runtime.DoAWorker", DoA)
    stop = threading.Event()
    stop.set()
    runtime = YRobotRuntime(
        FakeMini(events),
        Settings(realtime_url="ws://brain.local/v1/realtime?mode=video"),
        stop,
    )

    runtime.run()

    assert events == ["doa.stop"]


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
