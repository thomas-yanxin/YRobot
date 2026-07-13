"""Entry points.

* :class:`LiveChatApp` — the ``ReachyMiniApp`` the daemon discovers (entry point
  ``reachy_mini_apps``). It attaches the web UI to the framework's ``settings_app`` and
  runs the :class:`~reachy_mini_live_chat.pipeline.Pipeline` until the daemon stops it.
* :func:`cli` — a self-contained dev runner (``reachy-mini-live-chat --sim --stub``) that
  works without the daemon and even without the SDK installed.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time

from .config import Config

log = logging.getLogger("live_chat")

# The SDK base class is only needed when running under the daemon. Fall back to a shim
# so this module (and `cli`) import fine on a machine without `reachy_mini`.
try:
    from reachy_mini import ReachyMini, ReachyMiniApp  # type: ignore
    _HAS_SDK = True
except Exception:  # pragma: no cover - depends on environment
    _HAS_SDK = False

    class ReachyMiniApp:  # minimal shim
        custom_app_url = None
        dont_start_webserver = False
        request_media_backend = None
        settings_app = None

        def __init__(self, running_on_wireless: bool = False) -> None:
            self.running_on_wireless = running_on_wireless


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


class LiveChatApp(ReachyMiniApp):
    """Reachy Mini full-duplex live video chat."""

    def __init__(self, running_on_wireless: bool = False) -> None:
        super().__init__(running_on_wireless)
        self.cfg = Config.load()
        # Enable the framework web server (serves static/ + our routes) if configured.
        if self.cfg.web_ui:
            self.custom_app_url = f"http://0.0.0.0:{self.cfg.web_port}"
        self._pipeline = None

    def run(self, reachy_mini, stop_event) -> None:  # noqa: ANN001
        _setup_logging()
        from .pipeline import Pipeline

        pipeline = Pipeline(reachy_mini, self.cfg)
        self._pipeline = pipeline

        # attach web routes to the framework's FastAPI app, if present
        if self.cfg.web_ui and getattr(self, "settings_app", None) is not None:
            try:
                from .web import attach_routes

                attach_routes(self.settings_app, pipeline)
                log.info("web UI on %s", self.custom_app_url)
            except Exception as e:
                log.warning("web UI unavailable: %s", e)

        pipeline.start()
        try:
            while not stop_event.is_set() and not pipeline.bus.stop_event.is_set():
                time.sleep(0.1)
        finally:
            pipeline.shutdown()


# ---------------------------------------------------------------------------
# Standalone dev runner
# ---------------------------------------------------------------------------
def _make_mini(cfg: Config):
    if cfg.sim:
        from .sim import FakeMini

        log.info("using FakeMini (sim)")
        return FakeMini()
    if not _HAS_SDK:
        raise SystemExit("reachy_mini SDK not installed; use --sim for hardware-free dev.")
    log.info("connecting to Reachy Mini daemon...")
    return ReachyMini()


def _serve_web(pipeline, cfg: Config):
    """Start uvicorn in a daemon thread; return the Server (or None if unavailable)."""
    try:
        import uvicorn

        from .web import make_standalone_app
    except Exception as e:
        log.warning("web deps missing (%s); running headless (type to talk)", e)
        return None
    app = make_standalone_app(pipeline)
    config = uvicorn.Config(app, host="0.0.0.0", port=cfg.web_port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, name="uvicorn", daemon=True).start()
    log.info("web UI: http://localhost:%d", cfg.web_port)
    return server


def _stdin_loop(pipeline) -> None:
    """Headless fallback: read lines from stdin and inject them as turns."""
    print("Type a message and press Enter (Ctrl-D to quit):", flush=True)
    try:
        for line in sys.stdin:
            pipeline.inject_text(line.strip())
    except Exception:
        pass


def cli(argv=None) -> None:
    _setup_logging()
    parser = argparse.ArgumentParser(prog="reachy-mini-live-chat", description=LiveChatApp.__doc__)
    parser.add_argument("--sim", action="store_true", help="use the mock robot (no hardware/daemon)")
    parser.add_argument("--stub", action="store_true", help="stub engines (no ML models) — wiring/demo only")
    parser.add_argument("--no-web", action="store_true", help="disable the web UI (headless, stdin)")
    parser.add_argument("--lang", choices=["auto", "zh", "en"], help="force conversation language")
    parser.add_argument("--port", type=int, help="web UI port")
    args = parser.parse_args(argv)

    cfg = Config.load()
    cfg.sim = args.sim
    cfg.stub = args.stub
    if args.lang:
        cfg.lang = args.lang
    if args.port:
        cfg.web_port = args.port
    if args.no_web:
        cfg.web_ui = False

    from .pipeline import Pipeline

    mini = _make_mini(cfg)
    with mini:
        pipeline = Pipeline(mini, cfg)
        server = _serve_web(pipeline, cfg) if cfg.web_ui else None
        pipeline.start()
        try:
            if server is None:
                _stdin_loop(pipeline)
            else:
                while not pipeline.bus.stop_event.is_set():
                    time.sleep(0.2)
        except KeyboardInterrupt:
            print()
        finally:
            log.info("shutting down...")
            pipeline.shutdown()


if __name__ == "__main__":
    # When launched by the daemon: `python -u -m reachy_mini_live_chat.main`
    if _HAS_SDK and "--sim" not in sys.argv and "--stub" not in sys.argv:
        app = LiveChatApp()
        try:
            app.wrapped_run()
        except KeyboardInterrupt:
            app.stop()
    else:
        cli()
