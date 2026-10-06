#!/usr/bin/env python3
"""Replay recorded raw observations through the current stable manager, without YOLO.

This is a downstream regression/inspection tool, not a new detector inference run.
It reuses the video runner so every consecutive frame and annotation is processed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.tracking.person_tracker import PersonTracker  # noqa: E402
from src.tracking.stable_identity_manager import StableIdentityManager, StableIdentitySettings  # noqa: E402


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class RecordedTracker:
    """Runner adapter: consume raw observations only; never reuse recorded stable IDs."""
    processing_mode = 'cached_raw_observation_replay'
    annotate = staticmethod(PersonTracker.annotate)

    def __init__(self, raw_log: Path, settings: StableIdentitySettings, total_frames: int) -> None:
        self.identity_manager = StableIdentityManager(settings)
        self.total_frames = total_frames
        self.observations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        with raw_log.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                frame = int(row['frame_number'])
                if not 1 <= frame <= total_frames:
                    raise ValueError('Raw log frame outside recorded processed-frame range')
                self.observations[frame].append(dict(track_id=int(row['track_id']), confidence=float(row['confidence']),
                                                     **{key: int(row[key]) for key in ('x1', 'y1', 'x2', 'y2')}))
        self.frame_number = 0

    def reset(self) -> None:
        self.identity_manager.reset()
        self.frame_number = 0

    def track(self, frame: Any, timestamp_seconds: float | None = None) -> list[dict[str, Any]]:
        self.frame_number += 1
        if self.frame_number > self.total_frames:
            raise ValueError('Video is longer than source inference; refusing to invent raw observations')
        return self.identity_manager.update(frame, self.observations.get(self.frame_number, []), timestamp_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('body_tracking_runner', PROJECT_ROOT / 'scripts/08_test_body_tracking.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    source = args.source_run.resolve()
    log = source / 'reports' / f'{args.video.stem}_tracking_log.csv'
    stats = json.loads((source / 'reports' / f'{args.video.stem}_continuity_statistics.json').read_text())
    settings = StableIdentitySettings(**runner.load_settings()['stable_identity'])
    tracker = RecordedTracker(log, settings, int(stats['total_processed_frames']))
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run_config.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(processing_mode=tracker.processing_mode, source_run=str(source),
                       source_raw_log_sha256=file_hash(log), video_sha256=file_hash(args.video),
                       frame_limit=tracker.total_frames, stable_identity=asdict(settings),
                       source_sha256={str(path.relative_to(PROJECT_ROOT)): file_hash(path)
                                      for path in sorted((PROJECT_ROOT / "src/tracking").glob("*.py"))},
                       note='No YOLO inference performed. Processing FPS measures replay/encoding only.'), stream, indent=2)
    result = runner.process_video(args.video, tracker, output, False, max_frames=tracker.total_frames)
    if tracker.frame_number != tracker.total_frames:
        print('ERROR: Decoded frame count differs from source inference')
        return 1
    return 0 if result else 1


if __name__ == '__main__':
    raise SystemExit(main())
