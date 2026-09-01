from __future__ import annotations

import time
import cv2

from src.video.frame_sampler import interval_for


def process_camera(source, callback, mode: str = "realtime") -> None:
    capture = cv2.VideoCapture(source); interval = interval_for(mode); next_sample = 0.0
    try:
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok: break
            now = time.monotonic()
            if now >= next_sample: callback(frame); next_sample = now + interval
    finally: capture.release()
