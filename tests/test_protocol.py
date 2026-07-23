from __future__ import annotations

import base64
import json
import struct

import pytest

from yrobot.protocol import (
    INPUT_SAMPLE_RATE_HZ,
    OUTPUT_SAMPLE_RATE_HZ,
    Phase,
    ProtocolError,
    ProtocolState,
    ProtocolStateError,
    QueueDone,
    ResponseDelta,
    ServerError,
    SessionCreated,
    decode_output_audio,
    encode_input_audio,
    input_append,
    parse_server_event,
    serialize_client_event,
    session_close,
    session_init,
    transition_client,
    transition_server,
    validate_video_url,
)


def test_audio_is_16k_f32le_in_and_24k_f32le_out() -> None:
    samples = (-1.0, 0.0, 0.5, 1.0)
    pcm = struct.pack("<4f", *samples)

    assert INPUT_SAMPLE_RATE_HZ == 16_000
    assert base64.b64decode(encode_input_audio(pcm), validate=True) == pcm

    decoded = decode_output_audio(base64.b64encode(pcm).decode("ascii"))
    assert OUTPUT_SAMPLE_RATE_HZ == 24_000
    assert decoded.pcm_f32le == pcm
    assert decoded.sample_rate_hz == 24_000
    assert decoded.channels == 1
    assert decoded.sample_count == 4
    assert decoded.duration_seconds == 4 / 24_000


@pytest.mark.parametrize("pcm", [b"", b"\0", b"\0" * 5])
def test_audio_rejects_incomplete_f32le(pcm: bytes) -> None:
    with pytest.raises(ProtocolError):
        encode_input_audio(pcm)


def test_audio_and_video_reject_non_buffers() -> None:
    with pytest.raises(ProtocolError):
        encode_input_audio(4)  # type: ignore[arg-type]
    with pytest.raises(ProtocolError):
        input_append(struct.pack("<f", 0.0), video_frames=[4])  # type: ignore[list-item]


def test_exact_client_payloads_and_compact_json() -> None:
    ref = base64.b64encode(struct.pack("<f", 0.25)).decode("ascii")
    init = session_init(
        "Be concise.",
        length_penalty=1.1,
        ref_audio_base64=ref,
        tts_ref_audio_base64=ref,
    )
    assert init == {
        "type": "session.init",
        "payload": {
            "system_prompt": "Be concise.",
            "config": {"length_penalty": 1.1},
            "voice": {
                "ref_audio_base64": ref,
                "tts_ref_audio_base64": ref,
            },
        },
    }

    pcm = struct.pack("<2f", 0.0, 0.5)
    jpeg = b"\xff\xd8frame\xff\xd9"
    append = input_append(
        pcm,
        video_frames=(jpeg,),
        force_listen=True,
        max_slice_nums=1,
    )
    assert append == {
        "type": "input.append",
        "input": {
            "audio": base64.b64encode(pcm).decode("ascii"),
            "video_frames": [base64.b64encode(jpeg).decode("ascii")],
            "force_listen": True,
            "max_slice_nums": 1,
        },
    }
    assert session_close() == {"type": "session.close", "reason": "user_stop"}
    assert serialize_client_event(session_close()) == (
        '{"type":"session.close","reason":"user_stop"}'
    )


def test_url_must_be_exact_video_websocket_endpoint() -> None:
    url = "wss://minicpmo45.modelbest.cn/v1/realtime?mode=video"
    assert validate_video_url(url) == url

    invalid = (
        "https://minicpmo45.modelbest.cn/v1/realtime?mode=video",
        "wss://minicpmo45.modelbest.cn/v1/realtime",
        "wss://minicpmo45.modelbest.cn/v1/realtime?mode=audio",
        "wss://minicpmo45.modelbest.cn/v1/realtime?mode=video&mode=audio",
        "wss://minicpmo45.modelbest.cn/other?mode=video",
        "wss://[malformed/v1/realtime?mode=video",
        "wss://minicpmo45.modelbest.cn/v1/realtime?mode",
    )
    for candidate in invalid:
        with pytest.raises(ProtocolError):
            validate_video_url(candidate)


def test_server_events_are_typed_and_metrics_are_open() -> None:
    created = parse_server_event(
        json.dumps(
            {
                "type": "session.created",
                "session_id": "sess_1",
                "mode": "full_duplex",
                "metrics": {"backend": {"warm": True}, "new_metric": 7},
                "server_send_ts": 1_780_048_876.4,
            }
        )
    )
    assert isinstance(created, SessionCreated)
    assert created.metrics["new_metric"] == 7
    assert created.server_send_ts == 1_780_048_876.4

    audio = base64.b64encode(struct.pack("<2f", 0.1, -0.2)).decode("ascii")
    delta = parse_server_event(
        json.dumps(
            {
                "type": "response.output.delta",
                "kind": "audio",
                "session_id": "sess_1",
                "response_id": "resp_1",
                "audio": audio,
                "metrics": {"arbitrary_future_field": [1, 2]},
                "server_send_ts": 1_780_048_876.5,
            }
        )
    )
    assert isinstance(delta, ResponseDelta)
    assert delta.audio is not None
    assert delta.audio.sample_rate_hz == 24_000
    assert delta.server_send_ts == 1_780_048_876.5

    error = parse_server_event(
        json.dumps(
            {
                "type": "error",
                "error": {
                    "code": "queue_full",
                    "message": "Queue full",
                    "vendor_detail": {"retry": True},
                },
                "server_send_ts": 1_780_048_876.6,
            }
        )
    )
    assert isinstance(error, ServerError)
    assert error.error["vendor_detail"] == {"retry": True}
    assert error.server_send_ts == 1_780_048_876.6


def test_exact_lifecycle_queue_init_created_stream_close_closed() -> None:
    state = ProtocolState()
    state = transition_server(
        state,
        parse_server_event('{"type":"session.queue_done"}'),
    )
    assert state.phase is Phase.QUEUE_DONE

    state = transition_client(state, session_init("Be helpful."))
    assert state.phase is Phase.INIT

    created = parse_server_event(
        '{"type":"session.created","session_id":"sess_1","mode":"full_duplex","metrics":{}}'
    )
    state = transition_server(state, created)
    assert state.phase is Phase.CREATED
    assert state.session_id == "sess_1"

    state = transition_client(state, input_append(struct.pack("<f", 0.0)))
    assert state.phase is Phase.STREAMING

    listen = parse_server_event(
        '{"type":"response.output.delta","kind":"listen","session_id":"sess_1","metrics":{}}'
    )
    assert transition_server(state, listen) == state

    state = transition_client(state, session_close())
    assert state.phase is Phase.CLOSE
    state = transition_server(
        state,
        parse_server_event('{"type":"session.closed","session_id":"sess_1","reason":"user_stop"}'),
    )
    assert state.phase is Phase.CLOSED


def test_lifecycle_rejects_out_of_order_actions() -> None:
    with pytest.raises(ProtocolStateError):
        transition_client(ProtocolState(), session_init("too early"))
    with pytest.raises(ProtocolStateError):
        transition_server(
            ProtocolState(),
            SessionCreated("sess_1", "full_duplex", {}),
        )

    queued = transition_server(ProtocolState(), QueueDone())
    initialized = transition_client(queued, session_init("ok"))
    created = transition_server(
        initialized,
        SessionCreated("sess_1", "full_duplex", {}),
    )
    closed = transition_client(created, session_close())
    assert closed.phase is Phase.CLOSE


def test_server_can_close_during_initialization_or_before_first_input() -> None:
    queued = transition_server(ProtocolState(), QueueDone())
    initialized = transition_client(queued, session_init("ok"))
    timeout = parse_server_event(
        '{"type":"session.closed","session_id":"sess_late","reason":"timeout"}'
    )
    assert transition_server(initialized, timeout).phase is Phase.CLOSED

    created = transition_server(
        initialized,
        SessionCreated("sess_1", "full_duplex", {}),
    )
    close = transition_client(created, session_close())
    assert (
        transition_server(
            close,
            parse_server_event(
                '{"type":"session.closed","session_id":"sess_1","reason":"user_stop"}'
            ),
        ).phase
        is Phase.CLOSED
    )


def test_fatal_session_close_preserves_diagnostic() -> None:
    event = parse_server_event(
        '{"type":"session.closed","session_id":"sess_1",'
        '"reason":"backend_error","diagnostic":{"message":"GPU lost"},'
        '"server_send_ts":123.5}'
    )
    assert event.diagnostic == {"message": "GPU lost"}


@pytest.mark.parametrize(
    "raw",
    [
        '{"type":"response.audio.delta","delta":"AAAA"}',
        '{"type":"input_audio_buffer.speech_started"}',
        '{"type":"session.update","session":{}}',
        '{"type":"response.done"}',
        '{"type":"queue_done"}',
        '{"type":"session.queue_done","extra":true}',
        '{"type":"response.output.delta","kind":"audio","audio":"***"}',
        '{"type":"response.output.delta","kind":"text","text":1}',
        '{"type":"response.output.delta","kind":"listen","audio":"AAAA"}',
        '{"type":"session.created","session_id":"x","mode":"turn_based"}',
        '{"type":"session.queue_done","type":"session.queue_done"}',
        "[]",
    ],
)
def test_invalid_and_openai_style_server_events_are_rejected(raw: str) -> None:
    with pytest.raises(ProtocolError):
        parse_server_event(raw)
