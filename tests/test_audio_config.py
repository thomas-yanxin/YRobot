from __future__ import annotations

from types import SimpleNamespace

from yrobot.audio_config import AUDIO_STARTUP_CONFIG, apply_audio_startup_config


class FakeAudio:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.calls: list[tuple[object, bool, float]] = []

    def apply_audio_config(
        self,
        config: object,
        *,
        verify: bool,
        write_settle_seconds: float,
    ) -> bool:
        self.calls.append((config, verify, write_settle_seconds))
        return self.result


def test_official_full_duplex_audio_profile_is_applied_and_verified() -> None:
    audio = FakeAudio()
    mini = SimpleNamespace(media=SimpleNamespace(audio=audio))

    assert apply_audio_startup_config(mini)
    assert audio.calls == [(AUDIO_STARTUP_CONFIG, True, 0.1)]


def test_audio_profile_failure_is_a_safe_degradation() -> None:
    mini = SimpleNamespace(media=SimpleNamespace(audio=FakeAudio(False)))

    assert not apply_audio_startup_config(mini)
    assert not apply_audio_startup_config(object())
