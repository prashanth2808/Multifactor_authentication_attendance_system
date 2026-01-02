"""services/io_helpers.py

Helpers for cloud/web mode.

These functions MUST NOT access any hardware devices (no webcam, no microphone).
They only decode uploaded bytes into numpy arrays.
"""

from __future__ import annotations

import io
import wave
from typing import Tuple

import cv2
import numpy as np


def decode_image_bytes(image_bytes: bytes) -> np.ndarray | None:
    """Decode JPEG/PNG bytes to a BGR numpy image."""
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def decode_wav_bytes_to_float32(audio_bytes: bytes) -> Tuple[np.ndarray, int] | None:
    """Decode 16-bit PCM WAV bytes into float32 mono audio and sample rate."""
    if not audio_bytes:
        return None

    with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sampwidth != 2:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} (expected 16-bit PCM)")

    pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)

    return pcm, int(framerate)
