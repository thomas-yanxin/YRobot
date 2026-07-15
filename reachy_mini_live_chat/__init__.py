"""Reachy Mini — Realtime Full-Duplex Audio-Visual Dialogue.

A bilingual (zh/en) voice+video conversational app whose brain is a remote end-to-end
**omni** model (MiniCPM-o 4.5 via llama.cpp-omni) reached over a full-duplex WebSocket.
The robot side (Wireless CM4) streams 1 s mic chunks + a camera frame, plays the returned
speech, and drives DOA head-turns + expressive, safety-clamped motion.
"""

__version__ = "0.1.0"
