"""Unit tests for capture, VAD gating, echo guard, resampling and playback."""

import threading
import time

import numpy as np

from yrobot.audio import (
    FRAME_SAMPLES,
    EchoGuard,
    Microphone,
    Speaker,
    StreamResampler,
    UplinkGain,
    VoiceDetector,
)


class FakeVad:
    def __init__(self, result=True):
        self.result = result

    def is_speech(self, pcm, rate):
        return self.result


class FakeMedia:
    def __init__(self):
        self.pushed = []
        self.cleared = 0
        self.samples = []
        self.audio = self

    def get_audio_sample(self):
        return self.samples.pop(0) if self.samples else None

    def push_audio_sample(self, data):
        self.pushed.append(np.asarray(data))

    def clear_player(self):
        self.cleared += 1


def test_microphone_reframes_arbitrary_stereo_blocks():
    media = FakeMedia()
    media.samples = [np.zeros((450, 2), np.float32), np.zeros((200, 2), np.float32)]
    mic = Microphone(media)
    assert [len(f) for f in mic.read_frames()] == [FRAME_SAMPLES]
    assert [len(f) for f in mic.read_frames()] == [FRAME_SAMPLES]  # 130 carried over


def test_voice_detector_needs_streak_and_energy():
    det = VoiceDetector(vad=FakeVad(True))
    loud = np.full(FRAME_SAMPLES, 0.1, np.float32)
    quiet = np.full(FRAME_SAMPLES, 1e-4, np.float32)
    assert det.process(quiet, 0.00) is False  # energy below floor gate
    assert det.process(loud, 0.02) is False  # streak 1
    assert det.process(loud, 0.04) is False  # streak 2
    assert det.process(loud, 0.06) is True  # confirmed at 3
    assert det.active(0.30) is True
    assert det.active(0.40) is False


def test_voice_detector_adapts_noise_floor():
    det = VoiceDetector(vad=FakeVad(True))
    hum = np.full(FRAME_SAMPLES, 0.02, np.float32)  # steady motor noise
    for i in range(400):
        det.process(hum, i * 0.02)
    assert det.process(hum, 9.0) is False  # floor swallowed the hum
    speech = np.full(FRAME_SAMPLES, 0.3, np.float32)
    for i in range(3):
        det.process(speech, 10.0 + i * 0.02)
    assert det.process(speech, 10.06) is True


def test_voice_detector_frozen_floor_keeps_barge_sensitivity():
    det = VoiceDetector(vad=FakeVad(True))
    echo = np.full(FRAME_SAMPLES, 0.05, np.float32)
    # 10 s of the robot's own monologue echo: the floor must not learn it
    for i in range(500):
        det.process(echo, i * 0.02, floor_frozen=True)
    speech = np.full(FRAME_SAMPLES, 0.05, np.float32)  # user at the same level
    voiced = False
    for i in range(3):
        voiced = det.process(speech, 11.0 + i * 0.02, floor_frozen=True)
    assert voiced is True  # without the freeze the floor would gate this out


def test_echo_guard_decay_is_slow_and_bounded():
    guard = EchoGuard()
    for _ in range(500):  # 10 s of quiet playback frames: ~1 dB of decay
        guard.observe(mic_db=-80.0, playout_db=-10.0)
    assert guard.offset_db > EchoGuard.OFFSET_INIT_DB - 1.5
    for _ in range(10_000):  # minutes of quiet: clamped at the floor
        guard.observe(mic_db=-80.0, playout_db=-10.0)
    assert guard.offset_db == EchoGuard.OFFSET_MIN_DB


def test_uplink_gain_boosts_quiet_speech_not_noise():
    agc = UplinkGain()
    speech = np.full(8000, 0.03, np.float32)  # −30 dB: typical XVF capture
    first = agc.process(speech)
    assert float(np.abs(first).max()) > 0.03  # boosting immediately
    frozen = agc.gain
    agc.process(np.full(8000, 0.001, np.float32))  # room noise: no update
    assert agc.gain == frozen
    for _ in range(20):
        out = agc.process(speech)
    assert abs(float(np.sqrt(np.mean(np.square(out)))) - UplinkGain.TARGET_RMS) < 0.02


def test_uplink_gain_never_amplifies_loud_speech_or_clips():
    agc = UplinkGain()
    loud = np.full(8000, 0.5, np.float32)
    for _ in range(10):
        out = agc.process(loud)
    assert agc.gain == 1.0
    assert float(np.abs(out).max()) <= 1.0


def test_uplink_gain_releases_in_playback_without_confirmed_user():
    agc = UplinkGain()
    quiet_speech = np.full(8000, 0.02, np.float32)
    for _ in range(10):
        agc.process(quiet_speech)
    boosted = agc.gain
    assert boosted > 2.0
    for _ in range(5):
        agc.process(
            quiet_speech,
            playback_active=True,
            confirmed_user_voice=False,
        )
    assert 1.0 < agc.gain < boosted


def test_echo_guard_passes_when_nothing_played():
    assert EchoGuard().observe(mic_db=-40.0, playout_db=-120.0) is True


def test_echo_guard_blocks_predicted_residual():
    # leakage ratio -25 dB sits below the -18 dB initial prediction
    assert EchoGuard().observe(mic_db=-35.0, playout_db=-10.0) is False


def test_echo_guard_learns_leakage_from_false_triggers():
    guard = EchoGuard()
    # An onset transient (ratio -9 dB) pierces the initial prediction…
    assert guard.observe(mic_db=-19.0, playout_db=-10.0) is True
    # …a few verified false ducks teach it until the transient is gated…
    for _ in range(4):
        guard.penalize()
    assert guard.observe(mic_db=-19.0, playout_db=-10.0) is False
    # …while a user talking over the robot still passes.
    assert guard.observe(mic_db=-5.0, playout_db=-10.0) is True


def test_echo_guard_penalize_bumps_prediction():
    guard = EchoGuard()
    before = guard.offset_db
    guard.penalize()
    assert guard.offset_db == before + EchoGuard.FALSE_TRIGGER_BUMP_DB


def test_resampler_ratio_and_continuity():
    rs = StreamResampler(24_000, 16_000)
    ramp = np.linspace(0.0, 1.0, 24_000, dtype=np.float32)
    out = np.concatenate([rs.process(chunk) for chunk in np.array_split(ramp, 13)])
    assert abs(len(out) - 16_000) <= 2
    assert np.all(np.diff(out) >= 0)  # no seams between chunks


def test_speaker_plays_after_boundary_and_flushes_on_interrupt():
    media = FakeMedia()
    speaker = Speaker(media)
    speaker.start()
    try:
        speaker.play(speaker.epoch, np.ones(2400, np.float32))  # 100 ms < preroll
        speaker.utterance_end()  # boundary flushes the short reply out
        deadline = time.monotonic() + 2.0
        while not media.pushed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert media.pushed and len(media.pushed[0]) == 1600

        stale_epoch = speaker.epoch
        speaker.interrupt()
        speaker.play(stale_epoch, np.ones(24_000, np.float32))  # late, old turn
        deadline = time.monotonic() + 2.0
        while media.cleared == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert media.cleared == 1
        time.sleep(0.2)
        assert len(media.pushed) == 1  # stale audio never reached the device
    finally:
        speaker.close()
        speaker.join(timeout=2)


def test_speaker_duck_then_resume_is_lossless():
    media = FakeMedia()
    speaker = Speaker(media)
    speaker.start()
    try:
        speaker.play(speaker.epoch, np.ones(24_000, np.float32))  # 1 s turn
        deadline = time.monotonic() + 2.0
        while not media.pushed and time.monotonic() < deadline:
            time.sleep(0.01)
        assert media.pushed

        speaker.hold()
        assert speaker.hold_completed_at() is None
        deadline = time.monotonic() + 2.0
        while media.cleared == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert media.cleared == 1
        assert speaker.hold_completed_at() is not None
        assert speaker.audible()  # the held turn is still live
        assert speaker.playout_db(time.monotonic()) > -120.0  # envelope kept

        speaker.release_hold()
        deadline = time.monotonic() + 2.0
        while len(media.pushed) < 2 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(media.pushed) == 2
        assert 0 < len(media.pushed[1]) <= 16_000  # the un-played tail only
    finally:
        speaker.close()
        speaker.join(timeout=2)


def test_speaker_reports_clear_completion_not_request_time():
    clear_allowed = threading.Event()

    class BlockingClearMedia(FakeMedia):
        def clear_player(self):
            clear_allowed.wait(2.0)
            super().clear_player()

    media = BlockingClearMedia()
    speaker = Speaker(media)
    speaker.start()
    try:
        speaker.hold()
        time.sleep(0.08)
        assert speaker.hold_completed_at() is None
        clear_allowed.set()
        deadline = time.monotonic() + 2.0
        while speaker.hold_completed_at() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert speaker.hold_completed_at() is not None

        epoch = speaker.interrupt()
        assert speaker.wait_flushed(epoch, timeout=2.0)
        assert media.cleared >= 2
    finally:
        clear_allowed.set()
        speaker.close()
        speaker.join(timeout=2)
