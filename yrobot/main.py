"""Reachy app entry point and standalone CM4 runner."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import threading
from dataclasses import replace

from reachy_mini import ReachyMini, ReachyMiniApp

from .config import Config, normalize_backend_url
from .omni import OmniClient
from .robot import RobotIO

log = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def run_conversation(mini: object, config: Config, stop_event: threading.Event) -> None:
    robot = RobotIO(mini)
    client = OmniClient(config)
    try:
        robot.start()
        asyncio.run(client.run(robot, stop_event))
    finally:
        robot.stop()


class Yrobot(ReachyMiniApp):
    """Full-duplex MiniCPM-o conversation for Reachy Mini Wireless."""

    dont_start_webserver = True
    # YRobot's production target is the Wireless CM4 itself. Selecting LOCAL
    # explicitly avoids a costly WebRTC loopback if the SDK control connection
    # has to fall back from localhost to reachy-mini.local.
    request_media_backend = "local"

    def __init__(self, running_on_wireless: bool = False) -> None:
        super().__init__(running_on_wireless)
        self.config = Config.load()

    def run(self, reachy_mini: object, stop_event: threading.Event) -> None:
        configure_logging()
        run_conversation(reachy_mini, self.config, stop_event)


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=Yrobot.__doc__)
    parser.add_argument("--url", help="raw llama-omni-server WebSocket URL")
    parser.add_argument("--no-video", action="store_true", help="stream audio only")
    parser.add_argument(
        "--tls-verify", action="store_true", help="verify the Omni server certificate"
    )
    args = parser.parse_args(argv)

    configure_logging()
    config = Config.load()
    if args.url:
        config = replace(config, omni_url=normalize_backend_url(args.url))
    if args.no_video:
        config = replace(config, send_video=False)
    if args.tls_verify:
        config = replace(config, tls_verify=True)

    stop_event = threading.Event()
    try:
        with ReachyMini(automatic_body_yaw=True, media_backend="local") as mini:
            run_conversation(mini, config, stop_event)
    except KeyboardInterrupt:
        log.info("Stopping YRobot")
        stop_event.set()


if __name__ == "__main__":
    Yrobot().wrapped_run()
