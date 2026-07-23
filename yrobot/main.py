"""YRobot app — wiring between mic, camera, realtime session, speaker and motion.

Data paths (each arrow is one queue or callback, no polling in the hot path):
    mic ──(500 ms chunks + voice gate)──► OmniClient ──► gateway
    camera ──(JPEG, adaptive cadence)──┘
    gateway ──deltas──► TurnGate ──play──► speaker (flushable) ──► daemon wobble
Barge-in: voice onset while the robot speaks → flush player + ship the partial
chunk with force_listen; TurnGate discards the stale turn until the model
listens again (see yrobot.turn).
"""

from __future__ import annotations

import logging
import signal
import threading
import time
from typing import Optional

import numpy as np

from yrobot.audio import AudioIO, StreamResampler
from yrobot.config import Config
from yrobot.motion import Puppeteer
from yrobot.omni import OmniClient, ThinkFilter
from yrobot.state import Shared
from yrobot.turn import TurnGate

logger = logging.getLogger("yrobot")

try:
    from reachy_mini.apps.app import ReachyMiniApp
except ImportError:  # allow importing (e.g. tests) without the SDK
    class ReachyMiniApp:  # type: ignore[no-redef]
        def __init__(self, *a, **k):
            self.stop_event = threading.Event()

        def stop(self):
            self.stop_event.set()


class YRobotApp(ReachyMiniApp):
    """Reachy Mini app entry point (`reachy_mini_apps` group)."""

    def run(self, reachy_mini, stop_event: threading.Event) -> None:
        cfg = Config.from_env()
        logging.basicConfig(
            level=cfg.log_level,
            format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        Pipeline(cfg, reachy_mini).run_until(stop_event)


class Pipeline:
    def __init__(self, cfg: Config, mini):
        self.cfg = cfg
        self.mini = mini
        self.state = Shared()
        self.turn = TurnGate(cfg)
        self.resampler = StreamResampler(cfg.model_out_sr, 16000)
        self.audio = AudioIO(cfg, mini.media, self.state,
                             on_chunk=self._on_mic_chunk,
                             on_voice_edge=self._on_voice_edge)
        self.client = OmniClient(cfg, sink=self)
        self.motion: Optional[Puppeteer] = Puppeteer(mini, cfg, self.state) if cfg.motion else None
        self._last_frame_t = 0.0
        self._text: list[str] = []
        self._think = ThinkFilter()
        self._await_reply_since = 0.0  # end-of-user-speech timestamp for latency logs

    # -- lifecycle -------------------------------------------------------------

    def run_until(self, stop_event: threading.Event) -> None:
        mini, media = self.mini, self.mini.media
        media.start_recording()
        media.start_playing()
        try:
            mini.enable_wobbling()
        except Exception:
            logger.warning("head wobbling unavailable", exc_info=True)
        try:
            mini.wake_up()
        except Exception:
            logger.warning("wake_up failed", exc_info=True)

        self.audio.start()
        self.client.start()
        if self.motion:
            self.motion.start()
        logger.info("YRobot up — talk to me (chunk=%dms mode=%s video=%s)",
                    self.cfg.chunk_ms, self.cfg.effective_mode, self.cfg.send_video)
        try:
            stop_event.wait()
        finally:
            logger.info("shutting down")
            self.client.stop()
            if self.motion:
                self.motion.stop()
            self.audio.stop()
            for fn in (mini.disable_wobbling, mini.stop_head_tracking, mini.goto_sleep):
                try:
                    fn()
                except Exception:
                    pass
            media.stop_playing()
            media.stop_recording()

    # -- uplink ------------------------------------------------------------------

    def _on_voice_edge(self, active: bool) -> bool:
        """Mic thread; returns True to flush the partial chunk immediately."""
        now = time.monotonic()
        interrupt = self.turn.on_voice(active, now, self.state.robot_speaking())
        if interrupt:
            logger.info("barge-in: flushing playback, forcing listen")
            self.audio.clear_playback()
            self.resampler.reset()
        if not active:
            self._await_reply_since = now
        return interrupt

    def _on_mic_chunk(self, chunk: np.ndarray, is_flush: bool) -> None:
        frame = self._maybe_frame()
        force = self.turn.take_force_listen(time.monotonic()) or is_flush
        self.client.submit(chunk, frame, force)

    def _maybe_frame(self) -> Optional[bytes]:
        if not self.cfg.send_video:
            return None
        now = time.monotonic()
        # Vision costs ~64 kv-tokens per frame: full rate only around USER
        # speech — the robot doesn't need to watch itself talk.
        active = self.state.voice_active or now - self.state.last_voice_end < 3.0
        interval = self.cfg.frame_active_s if active else self.cfg.frame_idle_s
        if now - self._last_frame_t < interval:
            return None
        try:
            jpeg = self.mini.media.get_frame_jpeg()
        except Exception:
            return None
        if jpeg:
            self._last_frame_t = now
        return jpeg

    # -- downlink (OmniClient sink) ------------------------------------------------

    def on_ready(self, ready: bool) -> None:
        self.state.ready = ready
        if not ready:
            # Let any queued reply audio finish naturally — reconnecting takes
            # ~15 s anyway. Only a user barge-in flushes the player.
            self.turn.reset()
            self.resampler.reset()

    def on_listen(self) -> None:
        self.turn.on_listen(time.monotonic())
        if self._text:
            logger.info("🤖 %s", "".join(self._text))
            self._text.clear()

    def on_model_audio(self, pcm24k: np.ndarray) -> None:
        now = time.monotonic()
        if not self.turn.on_model_audio(now):
            return
        if self._await_reply_since:
            logger.info("first reply audio +%.2fs after user stopped",
                        now - self._await_reply_since)
            self._await_reply_since = 0.0
        self.audio.play(self.resampler.process(pcm24k))

    def on_text(self, text: str) -> None:
        text = self._think.feed(text)
        if text and not self.turn.latched():
            self._text.append(text)

    def quiet(self) -> bool:
        s = self.state
        return (not s.voice_active and not s.robot_speaking()
                and time.monotonic() - s.last_voice_end > 2.0)


def main() -> None:
    """Standalone entry point: run on the robot without the app manager."""
    app = YRobotApp()
    signal.signal(signal.SIGINT, lambda *_: app.stop())
    signal.signal(signal.SIGTERM, lambda *_: app.stop())
    app.wrapped_run()


if __name__ == "__main__":
    main()
