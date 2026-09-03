#!/usr/bin/env python3
"""Run independent YOLO + ByteTrack person tracking on videos or a webcam."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.person_tracker import PersonTracker  # noqa: E402

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
LOG_FIELDS = [
    "video_name", "frame_number", "timestamp_seconds", "track_id",
    "confidence", "x1", "y1", "x2", "y2",
]
SUMMARY_FIELDS = [
    "track_id", "first_frame", "last_frame", "frames_visible",
    "first_seen_seconds", "last_seen_seconds",
    "approximate_visible_duration_seconds",
]


def load_settings() -> dict:
    settings_path = PROJECT_ROOT / "config" / "settings.yaml"
    with settings_path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle).get("TRACKING", {})


def build_tracker(settings: dict) -> PersonTracker:
    model_setting = Path(settings.get("model", "yolo11n.pt"))
    model_path = model_setting if model_setting.is_absolute() else PROJECT_ROOT / model_setting
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return PersonTracker(
        model_path=model_path,
        confidence_threshold=float(settings.get("confidence_threshold", 0.40)),
        tracker_config=str(settings.get("tracker", "bytetrack.yaml")),
        person_class_id=int(settings.get("person_class_id", 0)),
        inference_size=int(settings.get("inference_size", 640)),
    )


def write_reports(video_stem: str, rows: list[dict], fps: float, reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    log_path = reports_dir / f"{video_stem}_tracking_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    observations: dict[int, list[int]] = defaultdict(list)
    for row in rows:
        observations[int(row["track_id"])].append(int(row["frame_number"]))
    summary_path = reports_dir / f"{video_stem}_track_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for track_id in sorted(observations):
            frames = observations[track_id]
            first_frame, last_frame = min(frames), max(frames)
            writer.writerow({
                "track_id": track_id,
                "first_frame": first_frame,
                "last_frame": last_frame,
                "frames_visible": len(frames),
                "first_seen_seconds": f"{(first_frame - 1) / fps:.3f}",
                "last_seen_seconds": f"{(last_frame - 1) / fps:.3f}",
                "approximate_visible_duration_seconds": f"{len(frames) / fps:.3f}",
            })
    print(f"Tracking log: {log_path.relative_to(PROJECT_ROOT)}")
    print(f"Track summary: {summary_path.relative_to(PROJECT_ROOT)}")


def process_video(
    video_path: Path, tracker: PersonTracker, output_root: Path, show_preview: bool,
) -> bool:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"WARNING: Cannot open video: {video_path}")
        return False
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        print(f"WARNING: Invalid FPS for {video_path.name}; using 30 FPS")
        fps = 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        print(f"WARNING: Invalid video dimensions: {video_path}")
        capture.release()
        return False

    videos_dir = output_root / "videos"
    frames_dir = output_root / "frames"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    output_path = videos_dir / f"{video_path.stem}_tracked.mp4"
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        print(f"ERROR: Could not create output video: {output_path}")
        capture.release()
        return False

    tracker.reset()
    rows: list[dict] = []
    frame_number = 0
    first_sample_saved = False
    middle_sample_saved = False
    final_tracked_frame = None
    final_tracked_number = 0
    print(f"\nVideo: {video_path.name}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            tracks = tracker.track(frame)
            annotated = tracker.annotate(frame, tracks)
            writer.write(annotated)

            active_ids = sorted(int(track["track_id"]) for track in tracks)
            if frame_number == 1 or frame_number % 100 == 0:
                print(f"Frame: {frame_number} | Active tracks: {active_ids}")
            timestamp = (frame_number - 1) / fps
            for track in tracks:
                rows.append({
                    "video_name": video_path.name,
                    "frame_number": frame_number,
                    "timestamp_seconds": f"{timestamp:.3f}",
                    **track,
                })

            if tracks:
                if not first_sample_saved:
                    cv2.imwrite(str(frames_dir / f"{video_path.stem}_first.jpg"), annotated)
                    first_sample_saved = True
                if not middle_sample_saved and total_frames and frame_number >= total_frames // 2:
                    cv2.imwrite(str(frames_dir / f"{video_path.stem}_middle.jpg"), annotated)
                    middle_sample_saved = True
                final_tracked_frame = annotated.copy()
                final_tracked_number = frame_number

            if show_preview:
                cv2.imshow("SmartPresence Body Tracking - press Q to stop", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    print("Stopped by user; the processed portion was saved.")
                    break
    except Exception as exc:
        print(f"ERROR while processing {video_path.name}: {exc}")
        return False
    finally:
        capture.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    if final_tracked_frame is not None:
        cv2.imwrite(
            str(frames_dir / f"{video_path.stem}_final_{final_tracked_number:06d}.jpg"),
            final_tracked_frame,
        )
    write_reports(video_path.stem, rows, fps, output_root / "reports")
    print(f"Output video: {output_path.relative_to(PROJECT_ROOT)}")
    print(f"Processed frames: {frame_number} | Tracking observations: {len(rows)}")
    if not rows:
        print("WARNING: No tracked people were found in this video.")
    return True


def process_camera(tracker: PersonTracker, camera_index: int) -> bool:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(f"ERROR: Could not open camera index {camera_index}")
        return False
    tracker.reset()
    print("Camera tracking started. Press Q to stop.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("WARNING: Camera stopped returning frames.")
                break
            annotated = tracker.annotate(frame, tracker.track(frame))
            cv2.imshow("SmartPresence Body Tracking - press Q to stop", annotated)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test YOLO + ByteTrack person tracking")
    parser.add_argument("--camera", action="store_true", help="track the default webcam")
    parser.add_argument("--camera-index", type=int, default=0)
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--preview", dest="show_preview", action="store_true")
    preview.add_argument("--no-preview", dest="show_preview", action="store_false")
    parser.set_defaults(show_preview=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    try:
        tracker = build_tracker(settings)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.camera:
        return 0 if process_camera(tracker, args.camera_index) else 1

    input_dir = PROJECT_ROOT / "test_data" / "videos" / "body_tracking"
    input_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    )
    if not videos:
        print(f"No supported videos found in: {input_dir.relative_to(PROJECT_ROOT)}")
        return 1
    show_preview = (
        bool(settings.get("show_preview", True))
        if args.show_preview is None else args.show_preview
    )
    output_root = PROJECT_ROOT / "outputs" / "tracking"
    succeeded = sum(process_video(path, tracker, output_root, show_preview) for path in videos)
    print(f"\nCompleted {succeeded}/{len(videos)} video(s).")
    return 0 if succeeded == len(videos) else 1


if __name__ == "__main__":
    raise SystemExit(main())
