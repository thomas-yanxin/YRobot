"""Unit tests for app-level helpers (no hardware, no network)."""

import queue
import threading
import time

import numpy as np
import pytest

import yrobot.main as main_module
from yrobot.config import Settings
from yrobot.main import FRAME_MAX_DIM, Conversation, LatestCamera, UplinkPacket, shrink_jpeg
from yrobot.realtime import Delta
from yrobot.turn import QUIET_S


def test_shrink_jpeg_downscales_to_model_vision_size():
    cv2 = pytest.importorskip("cv2")
    frame = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    jpeg = shrink_jpeg(frame)
    assert jpeg is not None
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    assert max(decoded.shape[:2]) == FRAME_MAX_DIM
    full = cv2.imencode(".jpg", frame)[1].tobytes()
    assert len(jpeg) < len(full) / 3  # meaningfully lighter on the uplink


def test_shrink_jpeg_keeps_small_frames():
    cv2 = pytest.importorskip("cv2")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    decoded = cv2.imdecode(np.frombuffer(shrink_jpeg(frame), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (240, 320)


def test_shrink_jpeg_none_frame():
    assert shrink_jpeg(None) is None


def test_urgent_uplink_discards_stale_backlog():
    packets: queue.Queue[UplinkPacket] = queue.Queue(maxsize=4)
    old = np.zeros(16_000, np.float32)
    for captured_at in (1.0, 2.0, 3.0):
        packets.put(
            UplinkPacket(
                old,
                force_listen=False,
                captured_at=captured_at,
                input_id=f"old-{captured_at}",
            )
        )
    urgent = UplinkPacket(
        np.ones(16_000, np.float32),
        force_listen=True,
        captured_at=4.0,
        input_id="forced",
    )
    Conversation._enqueue_packet(packets, urgent, flush_backlog=True)
    assert packets.qsize() == 1
    assert packets.get_nowait() is urgent


def test_latest_camera_keeps_only_newest_frame(monkeypatch):
    monkeypatch.setattr(main_module, "cv2", None)

    class FakeCameraMedia:
        def __init__(self):
            self.count = 0

        def get_frame_jpeg(self):
            self.count += 1
            return f"frame-{self.count}".encode()

    camera = LatestCamera(
        FakeCameraMedia(),
        active=lambda now: True,
        robot_audible=lambda now: False,
        active_period_s=0.02,
        idle_period_s=0.02,
    )
    camera.start()
    try:
        time.sleep(0.09)
        latest = camera.take_latest()
        assert latest is not None
        assert camera.take_latest() is None
        time.sleep(0.05)
        assert camera.take_latest() != latest
    finally:
        camera.close()
        camera.join(timeout=2)


def test_slow_websocket_sender_does_not_block_packet_producer():
    entered = threading.Event()
    release = threading.Event()

    class SlowClient:
        def send_chunk(self, audio, jpeg, force_listen, input_id):
            entered.set()
            release.wait(2.0)

    conversation = object.__new__(Conversation)
    conversation._session_dead = threading.Event()
    conversation._video_kv_est = 0.0
    conversation._turn_lock = threading.Lock()
    conversation._gate = main_module.TurnGate()
    packets: queue.Queue[UplinkPacket] = queue.Queue(maxsize=4)
    halt = threading.Event()
    packets.put(
        UplinkPacket(
            np.zeros(16_000, np.float32),
            False,
            time.monotonic(),
            "normal-1",
        )
    )
    sender = threading.Thread(
        target=conversation._send_loop,
        args=(SlowClient(), packets, halt, None),
    )
    sender.start()
    try:
        assert entered.wait(1.0)
        started = time.monotonic()
        Conversation._enqueue_packet(
            packets,
            UplinkPacket(
                np.ones(16_000, np.float32),
                False,
                time.monotonic(),
                "normal-2",
            ),
        )
        assert time.monotonic() - started < 0.05
    finally:
        halt.set()
        release.set()
        sender.join(timeout=2)


def _conversation_without_hardware() -> Conversation:
    class FakeMedia:
        pass

    class FakeMini:
        media = FakeMedia()

    return Conversation(
        Settings(head_tracking_weight=0.0),
        FakeMini(),
        threading.Event(),
    )


def test_barge_candidate_hard_stops_and_latches_force():
    conversation = _conversation_without_hardware()
    started = time.monotonic()
    old_epoch = conversation._speaker.epoch

    conversation._begin_barge(started)

    assert conversation._speaker.epoch == old_epoch + 1
    assert conversation._speaker._flush_event.is_set()
    assert conversation._gate.latched
    with conversation._turn_lock:
        assert conversation._gate.chunk_force_listen(started + 0.01)


def test_qualified_local_vad_interrupts_without_echo_level_gate():
    conversation = _conversation_without_hardware()

    class FakeMic:
        def read_frames(self):
            return [np.full(320, 0.01, np.float32)]

    class FakeDetector:
        streak = 5
        last_db = -40.0

        def process(self, frame, now, floor_frozen=False):
            return True

    class FakeSpeaker:
        epoch = 0
        interrupted = 0

        def sounding(self, now):
            return True

        def audible(self, now):
            return True

        def playing(self, now):
            return self.interrupted == 0

        def interrupt(self):
            self.interrupted += 1
            self.epoch += 1
            return self.epoch

    class FakeChoreo:
        def set_mode(self, mode):
            self.mode = mode

    conversation._mic = FakeMic()
    conversation._detector = FakeDetector()
    conversation._speaker = FakeSpeaker()
    conversation._choreo = FakeChoreo()

    conversation._process_mic()

    assert conversation._speaker.interrupted == 1
    assert conversation._gate.latched
    assert conversation._gate.force_pending


def test_interrupted_multi_branch_output_waits_for_force_listen_boundary():
    conversation = _conversation_without_hardware()
    pcm = np.ones(2400, np.float32)
    conversation._on_delta(Delta(kind="audio", audio=pcm, response_id="old"))
    conversation._speaker._q.get_nowait()

    started = time.monotonic()
    conversation._begin_barge(started)
    conversation._on_delta(
        Delta(kind="audio", audio=pcm, response_id="old-branch-2", received_at=started + 0.04)
    )
    conversation._on_delta(
        Delta(kind="audio", audio=pcm, response_id="old-branch-3", received_at=started + 0.06)
    )
    assert conversation._speaker._q.empty()

    with conversation._turn_lock:
        assert conversation._gate.chunk_force_listen(started + 0.08)
        conversation._gate.force_sent("forced-input", started + 0.08)
    conversation._on_delta(Delta(kind="listen", input_id="wrong-input", received_at=started + 0.09))
    assert conversation._gate.latched
    conversation._on_delta(
        Delta(kind="listen", input_id="forced-input", received_at=started + 0.10)
    )
    conversation._on_delta(
        Delta(
            kind="audio",
            audio=pcm,
            response_id="new-answer",
            received_at=started + QUIET_S + 0.10,
        )
    )
    epoch, queued = conversation._speaker._q.get_nowait()
    assert epoch == conversation._speaker.epoch
    assert queued is pcm
