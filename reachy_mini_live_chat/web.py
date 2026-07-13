"""FastAPI routes for the control/monitor web UI.

Exposes: an SSE event stream (state, transcript, latency, emotions), a text-injection
endpoint (type-to-talk), a camera-preview JPEG, and a state/config snapshot. Attached
either to the ReachyMiniApp's built-in ``settings_app`` (on-robot) or a standalone app
(``cli`` dev mode).
"""
from __future__ import annotations

import json
import logging
import queue
from typing import TYPE_CHECKING

log = logging.getLogger("live_chat.web")

if TYPE_CHECKING:
    from .pipeline import Pipeline


def attach_routes(app, pipeline: "Pipeline") -> None:
    from fastapi import Body
    from fastapi.responses import Response, StreamingResponse

    bus = pipeline.bus

    @app.get("/api/state")
    def state():
        return {
            "state": bus.state.value,
            "latency_ms": bus.last_latency_ms,
            "config": pipeline.cfg.as_dict(),
            "transcript": list(bus.transcript)[-50:],
        }

    @app.post("/api/say")
    def say(payload: dict = Body(...)):
        text = (payload or {}).get("text", "")
        pipeline.inject_text(text)
        return {"ok": True}

    @app.get("/api/frame.jpg")
    def frame():
        try:
            jpeg = pipeline.mini.media.get_frame_jpeg()
        except Exception:
            jpeg = None
        if not jpeg:
            return Response(status_code=204)
        return Response(content=jpeg, media_type="image/jpeg")

    @app.get("/api/events")
    def events():
        q: "queue.Queue[dict]" = queue.Queue(maxsize=200)
        bus.subscribe(lambda evt: _safe_put(q, evt))

        def gen():
            # replay recent transcript so a fresh page isn't blank
            for evt in list(bus.transcript)[-20:]:
                yield _sse(evt)
            yield _sse({"kind": "state", "state": bus.state.value})
            while not bus.stop_event.is_set():
                try:
                    evt = q.get(timeout=1.0)
                    yield _sse(evt)
                except queue.Empty:
                    yield ": keepalive\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")


def _safe_put(q: "queue.Queue", evt: dict) -> None:
    try:
        q.put_nowait(evt)
    except queue.Full:
        pass


def _sse(evt: dict) -> str:
    return f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"


def make_standalone_app(pipeline: "Pipeline"):
    """Build a standalone FastAPI app (dev/cli) serving static/ + the routes."""
    import os

    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="Reachy Mini Live Chat")
    static_dir = os.path.join(os.path.dirname(__file__), "static")

    attach_routes(app, pipeline)

    @app.get("/")
    def index():
        return FileResponse(os.path.join(static_dir, "index.html"))

    if os.path.isdir(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app
