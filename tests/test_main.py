import asyncio
import threading
from typing import Any

from yrobot.audio import PlayerClearError
from yrobot.config import Config
from yrobot.main import YRobotRuntime, app_main, cli


def make_config() -> Config:
    return Config(
        realtime_url="ws://127.0.0.1:8006/v1/realtime?mode=video",
        tls_verify=True,
        send_video=True,
        system_prompt="test",
    )


def test_runtime_starts_media_before_motion_and_stops_in_dependency_order(
    monkeypatch: Any,
) -> None:
    events: list[str] = []

    class Audio:
        def __init__(self, _mini: object, **kwargs: Any) -> None:
            self.state_callback = kwargs["state_callback"]
            self.error_callback = kwargs["error_callback"]
            self.metrics = {"ok": 1}

        def start(self, *, session_ready: bool) -> None:
            assert session_ready is False
            events.append("audio.start")

        def stop(self) -> None:
            events.append("audio.stop")

        def invalidate_session(self, _reason: str) -> None: ...

    class Motion:
        def __init__(self, _mini: object, **_kwargs: Any) -> None: ...

        def start(self) -> None:
            events.append("motion.start")

        def stop(self) -> None:
            events.append("motion.stop")

        def set_state(self, _state: object) -> None: ...

    class Client:
        def __init__(self, _config: Config) -> None: ...

        async def run(self, _port: object, stop_event: threading.Event) -> None:
            events.append("client.run")
            stop_event.set()
            await asyncio.sleep(0)

    monkeypatch.setattr("yrobot.main.AudioEngine", Audio)
    monkeypatch.setattr("yrobot.main.MotionController", Motion)
    monkeypatch.setattr("yrobot.main.RealtimeClient", Client)

    runtime = YRobotRuntime(object(), make_config(), threading.Event())
    runtime.run()

    assert events == [
        "audio.start",
        "motion.start",
        "client.run",
        "audio.stop",
        "motion.stop",
    ]


def test_player_clear_failure_stops_the_runtime(monkeypatch: Any) -> None:
    class Audio:
        def __init__(self, _mini: object, **_kwargs: Any) -> None:
            self.metrics: dict[str, int] = {}

    class Motion:
        def __init__(self, _mini: object, **_kwargs: Any) -> None: ...

    monkeypatch.setattr("yrobot.main.AudioEngine", Audio)
    monkeypatch.setattr("yrobot.main.MotionController", Motion)

    stop_event = threading.Event()
    runtime = YRobotRuntime(object(), make_config(), stop_event)
    runtime._handle_media_error(PlayerClearError("flush failed"))

    assert stop_event.is_set()


def test_cli_uses_wireless_local_media_path(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    class Mini:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

        def __enter__(self) -> "Mini":
            return self

        def __exit__(self, *_args: object) -> None: ...

    monkeypatch.setattr("yrobot.main.ReachyMini", Mini)
    monkeypatch.setattr("yrobot.main.Config.load", staticmethod(make_config))
    monkeypatch.setattr(
        "yrobot.main.run_conversation",
        lambda _mini, config, _stop, **kwargs: seen.update(
            config=config,
            runtime_options=kwargs,
        ),
    )

    cli(["--no-video", "--force-listen-count", "0"])

    assert seen["connection_mode"] == "localhost_only"
    assert seen["media_backend"] == "local"
    assert seen["automatic_body_yaw"] is True
    assert seen["config"].send_video is False
    assert seen["config"].force_listen_count == 0
    assert seen["runtime_options"] == {"neutral_transitions": True}


def test_dashboard_entry_uses_reachy_app_lifecycle(monkeypatch: Any) -> None:
    events: list[str] = []

    class App:
        def wrapped_run(self) -> None:
            events.append("wrapped_run")

        def stop(self) -> None:
            events.append("stop")

    monkeypatch.setattr("yrobot.main.Yrobot", App)
    app_main()

    assert events == ["wrapped_run"]
