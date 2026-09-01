from __future__ import annotations

from src.utils.paths import settings


def interval_for(mode: str) -> int:
    keys = {"experiment": "EXPERIMENT_FRAME_INTERVAL_SECONDS", "realtime": "REALTIME_CCTV_FRAME_INTERVAL_SECONDS"}
    if mode not in keys: raise ValueError("mode must be 'experiment' or 'realtime'")
    return int(settings()[keys[mode]])


def sampled_frames(video_path: str, mode: str = "experiment"):
    import cv2

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened(): raise FileNotFoundError(video_path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0; step = max(1, round(fps * interval_for(mode))); index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok: break
            if index % step == 0: yield index, index / fps, frame
            index += 1
    finally: capture.release()
