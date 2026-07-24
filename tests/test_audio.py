"""Unit tests for capture, VAD gating, resampling and playback pacing."""

import time

import numpy as np

from yrobot.audio import FRAME_SAMPLES, Microphone, Speaker, StreamResampler, VoiceDetector


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
