"""Exercise the live gateway with YRobot's own client — no robot needed.

Streams a 16 kHz mono WAV as paced mic chunks, prints reply latency and text,
and (with --barge-wav) simulates a barge-in mid-reply to check the turn gate.

    python scripts/probe_realtime.py --wav hello.wav [--barge-wav interrupt.wav]
"""

from __future__ import annotations

import argparse
import logging
import time
import wave

import numpy as np

from yrobot.config import Config
from yrobot.omni import OmniClient
from yrobot.turn import TurnGate

log = logging.getLogger("probe")


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, "need 16 kHz mono"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0).copy()


class ProbeSink:
    def __init__(self, cfg: Config):
        self.turn = TurnGate(cfg)
        self.ready = False
        self.speech_end_t = 0.0
        self.first_audio_dt: float | None = None
        self.played_s = 0.0
        self.dropped = 0
        self.text: list[str] = []

    def on_ready(self, ready: bool) -> None:
        self.ready = ready
        log.info("session %s", "ready" if ready else "down")

    def on_listen(self) -> None:
        self.turn.on_listen(time.monotonic())

    def on_model_audio(self, pcm24k: np.ndarray) -> None:
        if not self.turn.on_model_audio(time.monotonic()):
            self.dropped += 1
            return
        if self.first_audio_dt is None and self.speech_end_t:
            self.first_audio_dt = time.monotonic() - self.speech_end_t
            log.info("first reply audio +%.2fs after speech end", self.first_audio_dt)
        self.played_s += len(pcm24k) / 24000

    def on_text(self, text: str) -> None:
        self.text.append(text)

    def quiet(self) -> bool:
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--barge-wav")
    parser.add_argument("--url")
    args = parser.parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = Config.from_env()
    if args.url:
        cfg = Config(**{**cfg.__dict__, "url": args.url})
    sink = ProbeSink(cfg)
    client = OmniClient(cfg, sink)
    client.start()

    while not sink.ready:
        time.sleep(0.1)

    chunk = 16000 * cfg.chunk_ms // 1000
    speech = load_wav(args.wav)

    def send(samples: np.ndarray, voice: bool) -> None:
        for i in range(0, len(samples), chunk):
            sink.turn.on_voice(voice, time.monotonic(), robot_speaking=sink.played_s > 0)
            client.submit(samples[i:i + chunk], None, sink.turn.take_force_listen())
            time.sleep(cfg.chunk_ms / 1000)

    log.info("streaming %s (%.1fs)", args.wav, len(speech) / 16000)
    send(np.zeros(8000, np.float32), False)
    send(speech, True)
    sink.speech_end_t = time.monotonic()
    sink.turn.on_voice(False, sink.speech_end_t, robot_speaking=False)
    send(np.zeros(16000 * 6, np.float32), False)

    if args.barge_wav:
        barge = load_wav(args.barge_wav)
        log.info("BARGE-IN: streaming %s", args.barge_wav)
        t0 = time.monotonic()
        interrupted = sink.turn.on_voice(True, t0, robot_speaking=True)
        log.info("turn gate interrupted=%s", interrupted)
        send(barge, True)
        sink.turn.on_voice(False, time.monotonic(), robot_speaking=False)
        sink.speech_end_t, sink.first_audio_dt = time.monotonic(), None
        send(np.zeros(16000 * 8, np.float32), False)
        log.info("deltas dropped by turn gate: %d", sink.dropped)

    log.info("reply: %r (%.1fs audio played)", "".join(sink.text), sink.played_s)
    client.stop()


if __name__ == "__main__":
    main()
