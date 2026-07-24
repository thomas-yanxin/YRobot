"""Reachy Mini XVF3800 settings tuned for full-duplex conversation."""

from __future__ import annotations

import logging

AUDIO_STARTUP_CONFIG = (
    ("PP_AGCMAXGAIN", (10.0,)),
    ("PP_MIN_NS", (0.8,)),
    ("PP_MIN_NN", (0.8,)),
    ("PP_GAMMA_E", (0.5,)),
    ("PP_GAMMA_ETAIL", (0.5,)),
    ("PP_NLATTENONOFF", (0,)),
    ("PP_MGSCALE", (4.0, 1.0, 1.0)),
)


def apply_audio_startup_config(
    mini: object,
    *,
    logger: logging.Logger | None = None,
) -> bool:
    """Apply the official conversation app's AEC/noise-suppression profile."""

    log = logger or logging.getLogger(__name__)
    audio = getattr(getattr(mini, "media", None), "audio", None)
    apply_config = getattr(audio, "apply_audio_config", None)
    if not callable(apply_config):
        log.warning("Reachy audio config API is unavailable; using the current board settings")
        return False
    try:
        applied = bool(
            apply_config(
                AUDIO_STARTUP_CONFIG,
                verify=True,
                write_settle_seconds=0.1,
            )
        )
    except Exception:
        log.warning(
            "Reachy audio startup config failed; using the current board settings",
            exc_info=True,
        )
        return False
    if not applied:
        log.warning("Reachy audio startup config could not be verified")
        return False
    log.info("Reachy XVF3800 full-duplex audio config applied")
    return True
