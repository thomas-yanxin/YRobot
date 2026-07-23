"""Reachy Mini app and standalone Wireless entrypoints."""

from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import replace

from reachy_mini import ReachyMini, ReachyMiniApp

from .config import Settings
from .runtime import YRobotRuntime

log = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


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
        run_conversation(reachy_mini, settings, stop_event)


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
        with ReachyMini(
            connection_mode="localhost_only",
            automatic_body_yaw=True,
            media_backend="local",
        ) as mini:
            run_conversation(mini, settings, stop_event)
    except KeyboardInterrupt:
        stop_event.set()
        log.info("Stopping YRobot")


def app_main() -> None:
    app = Yrobot()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    app_main()
