"""End-to-end wiring smoke test: FakeMini + the real OmniClient against a fake omni
server — no GPU server, no hardware, no local ML models."""
import time

from reachy_mini_live_chat.config import Config
from reachy_mini_live_chat.omni.fake_server import serve_in_thread
from reachy_mini_live_chat.pipeline import Pipeline
from reachy_mini_live_chat.sim import FakeMini


def _cfg(port: int) -> Config:
    c = Config()
    c.sim = True
    c.stub = True
    c.web_ui = False
    c.enable_motion = True   # exercise the (procedural) 100 Hz control loop too
    c.enable_doa = False
    c.omni_send_video = False
    c.omni_ws_url = f"ws://127.0.0.1:{port}/backend"
    c.omni_out_sr = 24000
    return c


def test_omni_pipeline_produces_a_spoken_reply():
    # speak_every=2 → the fake server replies after ~2 one-second chunks
    _thread, stop, port = serve_in_thread(out_sr=24000, speak_every=2)
    try:
        cfg = _cfg(port)
        mini = FakeMini()
        pipeline = Pipeline(mini, cfg)
        pipeline.start()
        try:
            got_assistant = False
            got_audio = False
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                kinds = [e["kind"] for e in list(pipeline.bus.transcript)]
                got_assistant = kinds.count("assistant") >= 1
                # robot_speaking flips on when playback consumes an audio chunk
                got_audio = got_audio or pipeline.bus.robot_speaking.is_set()
                if got_assistant and got_audio:
                    break
                time.sleep(0.1)
            assert got_assistant, "omni brain never produced an assistant reply"
            assert got_audio, "no audio was played back"
        finally:
            pipeline.shutdown()
        assert pipeline.bus.stop_event.is_set()
    finally:
        stop()


def test_motion_controller_sets_target():
    _thread, stop, port = serve_in_thread(out_sr=24000, speak_every=99)
    try:
        cfg = _cfg(port)
        mini = FakeMini()
        pipeline = Pipeline(mini, cfg)
        pipeline.start()
        try:
            time.sleep(0.5)  # let the 100 Hz loop run
            head = mini.get_current_head_pose()
            assert head.shape == (4, 4)
        finally:
            pipeline.shutdown()
    finally:
        stop()
