"""End-to-end wiring smoke test: FakeMini + stub engines, no models, no hardware."""
import time

from reachy_mini_live_chat.config import Config
from reachy_mini_live_chat.pipeline import Pipeline
from reachy_mini_live_chat.sim import FakeMini


def _stub_cfg():
    c = Config()
    c.sim = True
    c.stub = True
    c.web_ui = False
    c.enable_vision = False
    c.enable_motion = True   # exercise the (procedural) control loop too
    c.enable_doa = False
    return c


def test_pipeline_handles_a_typed_turn():
    cfg = _stub_cfg()
    mini = FakeMini()
    pipeline = Pipeline(mini, cfg)
    pipeline.start()
    try:
        pipeline.inject_text("你好，你能听到我吗")
        # wait for the brain to produce an assistant reply
        got_user = got_assistant = False
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            kinds = [e["kind"] for e in list(pipeline.bus.transcript)]
            got_user = "user" in kinds
            got_assistant = kinds.count("assistant") >= 1
            if got_user and got_assistant:
                break
            time.sleep(0.05)
        assert got_user, "user turn was not recorded"
        assert got_assistant, "assistant never replied"
    finally:
        pipeline.shutdown()

    assert pipeline.bus.stop_event.is_set()


def test_motion_controller_sets_target():
    cfg = _stub_cfg()
    mini = FakeMini()
    pipeline = Pipeline(mini, cfg)
    pipeline.start()
    try:
        time.sleep(0.5)  # let the 100 Hz loop run
        # FakeMini stores last commanded pose; it should be a valid 4x4
        head = mini.get_current_head_pose()
        assert head.shape == (4, 4)
    finally:
        pipeline.shutdown()
