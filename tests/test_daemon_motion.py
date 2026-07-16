"""Daemon-native wobbler / face-tracking: feature detection + speak-time handoff."""
from reachy_mini_live_chat.bus import Bus
from reachy_mini_live_chat.config import Config
from reachy_mini_live_chat.motion.controller import MotionController


class _Media:
    def get_DoA(self):
        return None

    def push_audio_sample(self, x):
        pass


class _ModernMini:
    """SDK with daemon wobble + face tracking."""

    def __init__(self):
        self.media = _Media()
        self.wobbling = None
        self.tracking_calls = []

    def enable_motors(self):
        pass

    def enable_wobbling(self):
        self.wobbling = True

    def disable_wobbling(self):
        self.wobbling = False

    def start_head_tracking(self, weight=1.0):
        self.tracking_calls.append(weight)

    def stop_head_tracking(self):
        self.tracking_calls.append(None)


class _OldMini:
    """SDK without the daemon features."""

    def __init__(self):
        self.media = _Media()

    def enable_motors(self):
        pass


def test_modern_sdk_enables_daemon_features():
    cfg = Config()
    cfg.enable_face_tracking = True  # off by default (CPU/camera cost on the CM4)
    ctl = MotionController(_ModernMini(), cfg, Bus())
    ctl.start()
    assert ctl._daemon_wobble is True
    assert ctl._face_tracking is True
    assert ctl.mini.wobbling is True
    assert ctl.mini.tracking_calls == [1.0]
    ctl.bus.stop_event.set()
    ctl.join()
    assert ctl.mini.wobbling is False
    assert ctl.mini.tracking_calls[-1] is None  # stop_head_tracking on teardown


def test_old_sdk_falls_back_gracefully():
    ctl = MotionController(_OldMini(), Config(), Bus())
    ctl.start()
    assert ctl._daemon_wobble is False
    assert ctl._face_tracking is False
    ctl.bus.stop_event.set()
    ctl.join()


def test_config_can_disable_daemon_features():
    cfg = Config()
    cfg.enable_daemon_wobble = False
    cfg.enable_face_tracking = False
    ctl = MotionController(_ModernMini(), cfg, Bus())
    ctl.start()
    assert ctl._daemon_wobble is False
    assert ctl._face_tracking is False
    ctl.bus.stop_event.set()
    ctl.join()


def test_tracking_pauses_while_robot_speaks():
    ctl = MotionController(_ModernMini(), Config(), Bus())
    ctl._face_tracking = True
    ctl.bus.robot_speaking.set()
    ctl._update_tracking_pause()
    assert ctl.mini.tracking_calls[-1] == 0.0   # paused for the reply
    ctl._update_tracking_pause()                 # edge-triggered: no repeat
    assert ctl.mini.tracking_calls.count(0.0) == 1
    ctl.bus.robot_speaking.clear()
    ctl._update_tracking_pause()
    assert ctl.mini.tracking_calls[-1] == 1.0   # resumed after the reply


def test_daemon_wobble_stills_app_level_head():
    ctl = MotionController(_ModernMini(), Config(), Bus())
    ctl._daemon_wobble = True
    ctl._speech_lvl = 1.0
    for i in range(50):  # sweep t: random phases must never move the head
        (roll, pitch, yaw, z), ant = ctl._speaking(t=i * 0.05)
        assert roll == pitch == yaw == z == 0.0  # daemon owns the head wobble
    peak_ant = max(abs(ctl._speaking(t=i * 0.05)[1][0]) for i in range(50))
    assert peak_ant > 0.1                        # antennas still ours
