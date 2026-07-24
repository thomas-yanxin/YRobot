"""Reachy Mini app and standalone Wireless entrypoints."""

from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import replace
from logging.handlers import QueueHandler, QueueListener
from queue import SimpleQueue

from reachy_mini import ReachyMini, ReachyMiniApp

from .config import Settings
from .runtime import YRobotRuntime

log = logging.getLogger(__name__)
_logging_lock = threading.Lock()
_logging_listener: QueueListener | None = None
_logging_sinks: tuple[logging.Handler, ...] = ()
_logging_root_state: tuple[int, tuple[logging.Handler, ...]] | None = None
_logging_package_state: tuple[int, tuple[logging.Handler, ...], bool] | None = None


def configure_logging(level: str) -> None:
    """Keep log I/O off audio, realtime, DoA, and motion threads."""

    global _logging_listener, _logging_package_state, _logging_root_state, _logging_sinks
    with _logging_lock:
        root_logger = logging.getLogger()
        package_logger = logging.getLogger("yrobot")
        if _logging_listener is not None:
            root_logger.setLevel(level)
            package_logger.setLevel(level)
            return
        _logging_root_state = (root_logger.level, tuple(root_logger.handlers))
        _logging_package_state = (
            package_logger.level,
            tuple(package_logger.handlers),
            package_logger.propagate,
        )
        root_logger.setLevel(level)
        package_logger.setLevel(level)
        _logging_sinks = tuple(root_logger.handlers)
        if not _logging_sinks:
            sink = logging.StreamHandler()
            sink.setFormatter(
                logging.Formatter(
                    "%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S",
                )
            )
            _logging_sinks = (sink,)
        records: SimpleQueue[logging.LogRecord] = SimpleQueue()
        root_logger.handlers[:] = [QueueHandler(records)]
        package_logger.handlers.clear()
        package_logger.propagate = True
        _logging_listener = QueueListener(
            records,
            *_logging_sinks,
            respect_handler_level=True,
        )
        _logging_listener.start()


def shutdown_logging() -> None:
    """Drain queued diagnostics before the application exits."""

    global _logging_listener, _logging_package_state, _logging_root_state, _logging_sinks
    with _logging_lock:
        listener = _logging_listener
        sinks = _logging_sinks
        root_state = _logging_root_state
        package_state = _logging_package_state
        _logging_listener = None
        _logging_sinks = ()
        _logging_root_state = None
        _logging_package_state = None
    if listener is not None:
        listener.stop()
        root_logger = logging.getLogger()
        package_logger = logging.getLogger("yrobot")
        if root_state is not None:
            root_logger.setLevel(root_state[0])
            root_logger.handlers[:] = list(root_state[1])
        else:
            root_logger.handlers[:] = list(sinks)
        if package_state is not None:
            package_logger.setLevel(package_state[0])
            package_logger.handlers[:] = list(package_state[1])
            package_logger.propagate = package_state[2]


def run_conversation(
    mini: object,
    settings: Settings,
    stop_event: threading.Event,
) -> None:
    YRobotRuntime(mini, settings, stop_event).run()


class Yrobot(ReachyMiniApp):
    """MiniCPM-o 4.5 full-duplex conversation for Reachy Mini Wireless."""

    dont_start_webserver = True
    custom_app_url = None
    request_media_backend = "local"

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        settings = Settings.from_env()
        configure_logging(settings.log_level)
        try:
            run_conversation(reachy_mini, settings, stop_event)
        finally:
            shutdown_logging()


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=Yrobot.__doc__)
    parser.add_argument("--url", help="MiniCPM-o /v1/realtime?mode=video URL")
    parser.add_argument(
        "--tls-no-verify",
        action="store_true",
        help="accept a self-signed certificate on a trusted development gateway",
    )
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.url:
        settings = replace(settings, realtime_url=args.url)
    if args.tls_no_verify:
        settings = replace(settings, tls_verify=False)
    settings.validate()
    configure_logging(settings.log_level)

    stop_event = threading.Event()
    try:
        try:
            with ReachyMini(
                connection_mode="localhost_only",
                automatic_body_yaw=True,
                media_backend="local",
            ) as mini:
                run_conversation(mini, settings, stop_event)
        except KeyboardInterrupt:
            stop_event.set()
            log.info("Stopping YRobot")
    finally:
        shutdown_logging()


def app_main() -> None:
    app = Yrobot()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    app_main()
