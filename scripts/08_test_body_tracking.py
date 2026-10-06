#!/usr/bin/env python3
"""Run YOLO + ByteTrack/BoT-SORT with conservative stable identities on videos or a webcam."""

from __future__ import annotations

import argparse
import hashlib
import platform
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4
import sys
from pathlib import Path

import yaml

try:
    import cv2
except ImportError as exc:
    raise RuntimeError(
        "Missing dependency 'cv2' (opencv-python). "
        "Install project requirements first, for example:\n"
        "  python3 -m pip install -r requirements.txt"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.person_tracker import PersonTracker  # noqa: E402
from src.tracking.stable_identity_manager import StableIdentitySettings  # noqa: E402
from src.tracking.reporting import write_reports, LOG_FIELDS, SUMMARY_FIELDS  # noqa: E402

SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

TRACKER_MODES = {
    "bytetrack": "config/bytetrack_smartpresence.yaml",
    "botsort": "config/botsort_smartpresence.yaml",
}


def load_settings() -> dict:
    with (PROJECT_ROOT / "config/settings.yaml").open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    return {**document.get("TRACKING", {}), "stable_identity": document.get("STABLE_IDENTITY", {})}


def build_tracker(settings: dict) -> PersonTracker:
    model_setting = Path(settings.get("model", "models/tracking/yolo11s.pt"))
    model_path = model_setting if model_setting.is_absolute() else PROJECT_ROOT / model_setting
    if model_setting.parent == Path(".") and not model_path.exists():
        cached_model = PROJECT_ROOT / "models/tracking" / model_setting.name
        if cached_model.exists():
            model_path = cached_model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_config = str(settings.get("tracker", TRACKER_MODES["bytetrack"]))
    tracker_config = TRACKER_MODES.get(tracker_config, tracker_config)
    if (PROJECT_ROOT / tracker_config).is_file():
        tracker_config = str(PROJECT_ROOT / tracker_config)
    stable = dict(settings.get("stable_identity", {}))
    stable.setdefault("max_missing_frames", int(settings.get("stable_max_gap_frames", 120)))
    stable.setdefault("max_reassociation_gap_frames", int(settings.get("stable_max_gap_frames", 120)))
    return PersonTracker(
        model_path=model_path,
        confidence_threshold=float(settings.get("confidence_threshold", .10)),
        tracker_config=tracker_config, person_class_id=int(settings.get("person_class_id", 0)),
        inference_size=int(settings.get("inference_size", 960)),
        stable_settings=StableIdentitySettings(**stable),
        require_local_model=bool(settings.get("require_local_model", False)),
        cpu_threads=settings.get("cpu_threads"),
    )


def process_video(
    video_path: Path, tracker: PersonTracker, output_root: Path, show_preview: bool,
    max_frames: int | None = None,
) -> bool:
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"WARNING: Cannot open video: {video_path}")
        capture.release()
        return False
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
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
    if output_path.exists() or any((output_root / "reports").glob(f"{video_path.stem}_*")):
        capture.release()
        print(f"ERROR: Existing outputs for {video_path.stem}; choose a fresh --output-root")
        return False
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        print(f"ERROR: Could not create output video: {output_path}")
        writer.release()
        capture.release()
        return False

    tracker.reset()
    started = time.perf_counter()
    rows: list[dict] = []
    frame_number = 0
    first_sample_saved = False
    middle_sample_saved = False
    final_tracked_frame = None
    final_tracked_number = 0
    print(f"\nVideo: {video_path.name}")
    try:
        while max_frames is None or frame_number < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            tracks = tracker.track(frame, timestamp_seconds=(frame_number - 1) / fps)
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

    if max_frames is not None and frame_number == max_frames:
        print(f"Frame limit {max_frames} reached; this output covers a consecutive prefix only.")
    if final_tracked_frame is not None:
        cv2.imwrite(
            str(frames_dir / f"{video_path.stem}_final_{final_tracked_number:06d}.jpg"),
            final_tracked_frame,
        )
    write_reports(video_path.stem, rows, fps, output_root / "reports", total_frames=frame_number,
                  elapsed_seconds=time.perf_counter() - started, identities=tracker.identity_manager.states.values(),
                  processing_mode=getattr(tracker, "processing_mode", "full_inference"))
    print(f"Output video: {output_path}")
    print(f"Processed frames: {frame_number} | Tracking observations: {len(rows)}")
    if not rows:
        print("WARNING: No tracked people were found in this video.")
    return True


def process_camera(tracker: PersonTracker, camera_index: int, show_preview: bool = True) -> bool:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        print(f"ERROR: Could not open camera index {camera_index}")
        capture.release()
        return False
    tracker.reset()
    started = time.monotonic()
    print("Camera tracking started. Press Q to stop.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("WARNING: Camera stopped returning frames.")
                break
            annotated = tracker.annotate(frame, tracker.track(frame, timestamp_seconds=time.monotonic() - started))
            if show_preview:
                cv2.imshow("SmartPresence Body Tracking - press Q to stop", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break
    except Exception as exc:
        print(f"ERROR during camera tracking: {exc}")
        return False
    finally:
        capture.release()
        if show_preview:
            cv2.destroyAllWindows()
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test YOLO + ByteTrack/BoT-SORT person tracking")
    parser.add_argument("--camera", action="store_true", help="track the default webcam")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--video",
        type=Path,
        help="process one video file instead of discovering videos in the test directories",
    )
    parser.add_argument("--tracker", help="bytetrack, botsort, or a tracker YAML path")
    parser.add_argument("--model", help="YOLO weights path (e.g. models/tracking/yolo11s.pt)")
    parser.add_argument("--require-local-model", action="store_true", help="fail before detector construction if weights are missing; no detector download")
    parser.add_argument("--max-frames", type=int, help="recorded-video diagnostic limit; default processes the entire video")
    parser.add_argument("--cpu-threads", type=int, help="optional PyTorch CPU thread limit; useful on constrained CPUs")
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--disable-stable-id", action="store_true")
    parser.add_argument("--disable-appearance", action="store_true", help="disable SmartPresence appearance, not backend BoT-SORT ReID")
    parser.add_argument("--output-root", type=Path, help="fresh output directory; existing video/reports are never overwritten")
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--preview", dest="show_preview", action="store_true")
    preview.add_argument("--no-preview", dest="show_preview", action="store_false")
    parser.set_defaults(show_preview=None)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if args.max_frames is not None and args.max_frames < 1:
        print("ERROR: --max-frames must be positive")
        return 1
    settings = load_settings()
    for arg, key in (("tracker", "tracker"), ("model", "model"), ("imgsz", "inference_size"), ("conf", "confidence_threshold")):
        value = getattr(args, arg)
        if value is not None:
            settings[key] = value
    if args.cpu_threads is not None:
        settings["cpu_threads"] = args.cpu_threads
    if args.require_local_model:
        settings["require_local_model"] = True
    if args.disable_stable_id:
        settings["stable_identity"]["enabled"] = False
    if args.disable_appearance:
        settings["stable_identity"]["appearance_enabled"] = False
    try:
        tracker = build_tracker(settings)
    except (RuntimeError, ValueError, TypeError, OSError) as exc:
        print(f"ERROR: {exc}")
        return 1
    if args.camera:
        return 0 if process_camera(tracker, args.camera_index, args.show_preview is not False) else 1

    if args.video is not None:
        video = args.video if args.video.is_absolute() else PROJECT_ROOT / args.video
        videos = [video.resolve()] if video.is_file() else []
        input_dir = video.parent
    else:
        input_dirs = [
            PROJECT_ROOT / "test_data" / "videos" / "body_tracking",
            PROJECT_ROOT / "dataset" / "test_videos",
        ]
        for candidate in input_dirs:
            candidate.mkdir(parents=True, exist_ok=True)
        videos = sorted({
            path.resolve()
            for input_dir in input_dirs
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        })
        input_dir = input_dirs[0]
    if not videos:
        print(f"No supported videos found for: {args.video or input_dir.relative_to(PROJECT_ROOT)}")
        return 1
    show_preview = (
        bool(settings.get("show_preview", True))
        if args.show_preview is None else args.show_preview
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output_root = args.output_root or PROJECT_ROOT / "outputs/tracking/runs" / run_id
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_config.json"
    if manifest_path.exists():
        print("ERROR: run_config.json exists; choose a fresh --output-root")
        return 1
    import ultralytics
    config_path = Path(tracker.tracker_config)
    if not config_path.is_file():
        config_path = Path(ultralytics.__file__).parent / "cfg/trackers" / tracker.tracker_config
    source_files = [Path(__file__), *sorted((PROJECT_ROOT / "src/tracking").glob("*.py"))]
    source_hashes = {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files}
    import torch
    manifest = dict(processing_mode="full_inference", frame_limit=args.max_frames, cpu_threads=tracker.cpu_threads, python_version=platform.python_version(),
                    torch_version=torch.__version__, opencv_version=cv2.__version__,
                    cuda_available=torch.cuda.is_available(), source_sha256=source_hashes,
                    model_sha256=hashlib.sha256(Path(tracker.model_path).read_bytes()).hexdigest()
                    if Path(tracker.model_path).is_file() else None,
                    input_video_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in videos},
                    ultralytics_version=ultralytics.__version__, model=tracker.model_path,
                    tracker=tracker.tracker_config, confidence=tracker.confidence_threshold,
                    inference_size=tracker.inference_size, stable_identity=asdict(tracker.identity_manager.settings),
                    tracker_yaml=config_path.read_text() if config_path.is_file() else None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    succeeded = sum(process_video(path, tracker, output_root, show_preview, max_frames=args.max_frames) for path in videos)
    print(f"\nCompleted {succeeded}/{len(videos)} video(s).")
    return 0 if succeeded == len(videos) else 1


if __name__ == "__main__":
    raise SystemExit(main())
