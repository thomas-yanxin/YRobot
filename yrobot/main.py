"""Reachy Mini application entry point and standalone Wireless CM4 runner."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from reachy_mini import ReachyMini, ReachyMiniApp

from .audio import AudioEngine, PlayerClearError
from .config import Config, normalize_realtime_url
from .motion import MotionController, MotionState
from .realtime import RealtimeClient

log = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure compact logs once for dashboard and CLI execution."""

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class _ConversationPort:
    """Keep transcript logging separate from real-time media ownership."""

    def __init__(self, audio: AudioEngine) -> None:
        self.audio = audio
        self._text_chunks: list[str] = []

    def next_audio_unit(self, timeout: float) -> tuple[np.ndarray, bool] | None:
        return self.audio.next_audio_unit(timeout)

    def latest_frame_jpeg(self) -> bytes | None:
        return self.audio.latest_frame_jpeg()

    def handle_audio_delta(
        self,
        samples: np.ndarray,
        response_id: str,
        metrics: Mapping[str, Any],
    ) -> None:
        self.audio.handle_audio_delta(samples, response_id, metrics)

    def handle_listen(
        self,
        response_id: str,
        metrics: Mapping[str, Any],
    ) -> None:
        self.audio.handle_listen(response_id, metrics)
        if self._text_chunks:
            log.info("MiniCPM-o: %s", "".join(self._text_chunks).strip())
            self._text_chunks.clear()

    def handle_text(self, text: str, response_id: str) -> None:
        del response_id
        self._text_chunks.append(text)

    def handle_session_ready(self) -> None:
        self.audio.handle_session_ready()

    def invalidate_session(self, reason: str) -> None:
        self._text_chunks.clear()
        self.audio.invalidate_session(reason)

    def ready_for_rollover(self) -> bool:
        return self.audio.ready_for_rollover()


class YRobotRuntime:
    """Own one conversation and shut every subsystem down in dependency order."""

    def __init__(
        self,
        mini: object,
        config: Config,
        stop_event: threading.Event,
        *,
        neutral_transitions: bool = False,
    ) -> None:
        self.config = config
        self.stop_event = stop_event
        self.motion = MotionController(mini, neutral_transitions=neutral_transitions)
        self.audio = AudioEngine(
            mini,
            capture_video=config.send_video,
            uplink_unit_samples=config.audio_unit_samples,
            camera_fps=1.0 / config.frame_active_interval,
            camera_idle_fps=1.0 / config.frame_idle_interval,
            playback_lead_seconds=config.playback_lead_seconds,
            state_callback=self._handle_media_state,
            error_callback=self._handle_media_error,
        )
        self.port = _ConversationPort(self.audio)
        self.client = RealtimeClient(config)

    def run(self) -> None:
        """Run until the dashboard, Ctrl-C, or a fail-safe requests stop."""

        if self.stop_event.is_set():
            return

        audio_started = False
        try:
            log.info(
                "YRobot realtime: %s, audio=%d ms, video=%s, rollover=%.0f s",
                self.config.realtime_url,
                self.config.audio_chunk_ms,
                self.config.send_video,
                self.config.session_rollover,
            )
            # Starting media first activates the XVF far-end reference. The
            # AudioEngine then applies verified AEC tuning before DoA USB reads
            # begin in the motion subsystem.
            self.audio.start(session_ready=False)
            audio_started = True
            self.motion.start()
            asyncio.run(self.client.run(self.port, self.stop_event))
        finally:
            # RealtimeClient invalidates its session before returning. Stop the
            # speaker/capture path next, then the only motor owner and wobbling.
            if audio_started:
                self.audio.stop()
            self.motion.stop()
            log.info("YRobot media metrics: %s", self.audio.metrics)

    def _handle_media_state(self, state: str) -> None:
        try:
            self.motion.set_state(MotionState(state))
        except ValueError:
            log.warning("Ignoring unknown media state: %s", state)

    def _handle_media_error(self, error: BaseException) -> None:
        if isinstance(error, PlayerClearError):
            # Continuing would violate the guarantee that interrupted speech
            # can never resume from a device-side queue.
            log.critical("Cannot guarantee speaker silence; stopping YRobot: %s", error)
            self.stop_event.set()


def run_conversation(
    mini: object,
    config: Config,
    stop_event: threading.Event,
    *,
    neutral_transitions: bool = False,
) -> None:
    """Run a complete YRobot lifecycle on an already connected robot."""

    YRobotRuntime(
        mini,
        config,
        stop_event,
        neutral_transitions=neutral_transitions,
    ).run()


class Yrobot(ReachyMiniApp):
    """Full-duplex MiniCPM-o 4.5 conversation for Reachy Mini Wireless."""

    dont_start_webserver = True
    request_media_backend = "local"

    def __init__(self, running_on_wireless: bool = False) -> None:
        super().__init__(running_on_wireless)
        self.config = Config.load()

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        configure_logging()
        run_conversation(reachy_mini, self.config, stop_event)


def _force_listen_count(value: str) -> int:
    parsed = int(value)
    if not 0 <= parsed <= 10:
        raise argparse.ArgumentTypeError("must be between 0 and 10")
    return parsed


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=Yrobot.__doc__)
    parser.add_argument("--url", help="MiniCPM-o Realtime Gateway URL")
    parser.add_argument("--no-video", action="store_true", help="send microphone audio only")
    parser.add_argument(
        "--tls-no-verify",
        action="store_true",
        help="disable certificate verification for a wss:// development proxy",
    )
    parser.add_argument(
        "--force-listen-count",
        type=_force_listen_count,
        help="startup listen units (recommended: 1)",
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = Config.load()
    if args.url:
        config = replace(config, realtime_url=normalize_realtime_url(args.url))
    if args.no_video:
        config = replace(config, send_video=False)
    if args.tls_no_verify:
        config = replace(config, tls_verify=False)
    if args.force_listen_count is not None:
        config = replace(config, force_listen_count=args.force_listen_count)

    stop_event = threading.Event()
    try:
        with ReachyMini(
            connection_mode="localhost_only",
            automatic_body_yaw=True,
            media_backend="local",
        ) as mini:
            run_conversation(
                mini,
                config,
                stop_event,
                neutral_transitions=True,
            )
    except KeyboardInterrupt:
        log.info("Stopping YRobot")
        stop_event.set()


def app_main() -> None:
    """Entrypoint used by the Reachy dashboard app manager."""

    app = Yrobot()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    app_main()
