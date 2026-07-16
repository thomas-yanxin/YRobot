"""Tune the ReSpeaker XVF3800 audio board at startup — the way the official app does.

Reachy Mini's mic is a Seeed reSpeaker 4-mic array built on the XMOS XVF3800. Acoustic
echo cancellation, noise suppression and auto-gain all run *in that chip's firmware and are
always on*, so the audio the SDK hands us (``media.get_audio_sample()``) already has the
robot's own speaker removed. The right place to fight echo is therefore the chip, not our
Python — this is exactly what Pollen's ``reachy_mini_conversation_app`` does: at startup it
writes a small set of tuned post-processing parameters over the SDK
(``media.audio.apply_audio_config``), then relies on the hardware and streams the mic
continuously.

We mirror their tuned values. In particular ``PP_GAMMA_E`` / ``PP_GAMMA_ETAIL`` raise the
residual-echo suppression (the leftover tail that a fixed AEC lets through — the thing that
was making our robot hear itself and cut its own speech), and ``PP_AGCMAXGAIN`` is the
hardware auto-gain, so no software gain is needed.

Best-effort: if the board or the SDK API isn't present (e.g. an older SDK),
we log and carry on — the app still works, just without the tuning.
"""
from __future__ import annotations

import logging

log = logging.getLogger("live_chat.respeaker")

# (parameter_name, values) pairs, matching Pollen's reachy_mini_conversation_app defaults.
AUDIO_STARTUP_CONFIG = (
    ("PP_AGCMAXGAIN", (10.0,)),      # hardware auto-gain ceiling (replaces any software gain)
    ("PP_MIN_NS", (0.8,)),           # noise-suppression floor
    ("PP_MIN_NN", (0.8,)),
    ("PP_GAMMA_E", (0.5,)),          # residual echo suppression
    ("PP_GAMMA_ETAIL", (0.5,)),      # residual echo *tail* suppression (kills the self-echo tail)
    ("PP_NLATTENONOFF", (0,)),       # non-linear attenuation off
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
)


def apply_startup_config(mini, *, verify: bool = True) -> bool:
    """Write the tuned XVF3800 config to the ReSpeaker. Returns True on success.

    Never raises: a missing board / SDK API just logs and returns False.
    """
    audio = getattr(getattr(mini, "media", None), "audio", None)
    if audio is None:
        log.info("respeaker: no media.audio — skipping XVF3800 tuning")
        return False
    apply = getattr(audio, "apply_audio_config", None)
    if not callable(apply):
        log.warning("respeaker: SDK has no apply_audio_config — update reachy-mini to tune the "
                    "XVF3800 (echo/NS/AGC stay at chip defaults)")
        return False
    try:
        ok = bool(apply(AUDIO_STARTUP_CONFIG, verify=verify))
    except Exception as e:
        log.warning("respeaker: apply_audio_config failed (%s) — using chip defaults", e)
        return False
    if ok:
        log.info("respeaker: XVF3800 tuned (hardware AEC+NS+AGC) — %s",
                 ", ".join(f"{n}={' '.join(map(str, v))}" for n, v in AUDIO_STARTUP_CONFIG))
    else:
        log.warning("respeaker: XVF3800 config not applied (board unavailable?)")
    return ok
