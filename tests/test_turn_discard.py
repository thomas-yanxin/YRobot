"""A barge-in must cancel the WHOLE reply, not just audio while the user talks.

The omni server streams long replies in bursts (10+ s can be in flight), and
bus.interrupt_event clears as soon as the user stops talking — so the pipeline
keeps a per-turn discard latch: from the interrupt until the server's next turn
boundary (listen/done), every text/audio event of the old reply is dropped.
"""
import numpy as np

from reachy_mini_live_chat.bus import Bus
from reachy_mini_live_chat.config import Config
from reachy_mini_live_chat.pipeline import Pipeline


def _pipeline():
    # Sink-logic only: build the object without __init__ (no hardware, no threads).
    p = Pipeline.__new__(Pipeline)
    p.cfg = Config()
    p.cfg.enable_motion = False
    p.bus = Bus()
    p._speaking = False
    p._turn_text = ""
    p._discard_turn = False
    p.bus.subscribe(p._on_bus_event)
    return p


def _audio():
    return np.ones(1000, dtype=np.float32)


def _queued_audio(bus):
    """Count real audio arrays in tts_audio (ignoring None flush sentinels)."""
    items = []
    while bus.tts_audio.qsize():
        items.append(bus.tts_audio.get_nowait())
    return sum(1 for x in items if x is not None)


def test_barge_discards_rest_of_reply_until_listen():
    p = _pipeline()
    p.on_audio(_audio())
    assert p.bus.tts_audio.qsize() == 1  # reply playing normally

    p.bus.request_interrupt()            # user barges in (drains the queue too)
    assert p.bus.tts_audio.qsize() == 0

    p.bus.clear_interrupt()              # user stopped talking — interrupt flag drops
    p.on_audio(_audio())                 # ...but the OLD reply is still streaming in
    p.on_text("leftover of the old reply")
    assert p.bus.tts_audio.qsize() == 0, "old-reply audio must stay dead after the flag clears"

    p.on_listen()                        # server yields the floor: turn boundary
    p.on_audio(_audio())                 # next reply plays normally
    assert _queued_audio(p.bus) == 1


def test_turn_done_also_ends_discard_without_gesture():
    p = _pipeline()
    p.on_audio(_audio())
    p.bus.request_interrupt()
    p.bus.clear_interrupt()

    p.on_turn_done("full text of the reply that was cut off")
    kinds = []
    while p.bus.motion_intents.qsize():
        kinds.append(p.bus.motion_intents.get_nowait().kind)
    assert "emotion" not in kinds, "no gesture for words never spoken"

    p.on_audio(_audio())                 # next reply unaffected
    assert _queued_audio(p.bus) == 1


def test_interrupted_text_stays_out_of_transcript():
    p = _pipeline()
    p.on_text("first half")
    p.bus.request_interrupt()
    p.bus.clear_interrupt()
    p.on_text("second half that was never heard")
    shown = [e["text"] for e in p.bus.transcript if e["kind"] == "assistant"]
    assert shown == ["first half"]


def test_mid_barge_listen_does_not_unlatch_discard():
    """This omni server emits listen/done every ~1 s step. Forced listen steps that
    arrive WHILE the user is still barging must not clear the discard latch, or the
    old reply resumes right after ("pauses, then keeps talking")."""
    p = _pipeline()
    p.on_audio(_audio())
    p.bus.request_interrupt()            # user starts talking over the robot

    p.on_listen()                        # forced listen step, mid-barge
    p.on_turn_done("step boundary")      # step done, mid-barge
    p.on_audio(_audio())                 # old reply still streaming
    p.on_text("old reply leftover")
    assert _queued_audio(p.bus) == 0, "audio must stay dead while the barge is live"

    p.bus.clear_interrupt()              # user stopped talking
    p.on_audio(_audio())                 # still the old reply: discard holds...
    assert _queued_audio(p.bus) == 0
    p.on_listen()                        # ...until a boundary AFTER the barge ended
    p.on_audio(_audio())
    assert _queued_audio(p.bus) == 1     # fresh reply plays


def test_audio_dropped_while_interrupt_active_even_without_latch():
    p = _pipeline()
    p.bus.interrupt_event.set()          # active barge, latch not set
    p.on_audio(_audio())
    p.on_text("should not appear")
    assert p.bus.tts_audio.qsize() == 0
    assert [e for e in p.bus.transcript if e["kind"] == "assistant"] == []


def test_new_reply_does_not_clear_active_interrupt():
    p = _pipeline()
    p.bus.request_interrupt()
    p.on_audio(_audio())                 # dropped; must NOT clear the interrupt
    assert p.bus.interrupt_event.is_set()
