"""Descriptive tracking reports; missing frames never count as visible presence."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from src.tracking.stable_identity_manager import StableIdentity

LOG_FIELDS = ['video_name', 'frame_number', 'timestamp_seconds', 'track_id', 'stable_id',
              'roll_number', 'session_id', 'confidence', 'x1', 'y1', 'x2', 'y2']
TIME_FIELDS = ['first_frame', 'last_frame', 'frames_visible', 'first_seen_seconds',
               'last_seen_seconds', 'approximate_visible_duration_seconds']
SUMMARY_FIELDS = ['track_id', *TIME_FIELDS]
STABLE_FIELDS = ['stable_id', 'roll_number', 'session_id', 'raw_track_ids', *TIME_FIELDS,
                 'number_of_raw_id_changes', 'final_state']
DIAGNOSTIC_FIELDS = ['stable_id', 'raw_track_count', 'raw_id_changes', 'longest_gap_frames',
                     'reassociation_count', 'total_visible_frames', 'start_frame', 'end_frame', 'roll_number']
DISCLAIMER = 'Tracking continuity statistics are descriptive and do not represent IDF1, HOTA, MOTA, or identity accuracy.'


def _write_csv(destination: Path, fields: list[str], rows: Iterable[dict]) -> None:
    with destination.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _times(frames: list[int], fps: float) -> dict[str, Any]:
    frames = sorted(set(frames))
    return dict(first_frame=frames[0], last_frame=frames[-1], frames_visible=len(frames),
                first_seen_seconds=f'{(frames[0]-1)/fps:.3f}', last_seen_seconds=f'{(frames[-1]-1)/fps:.3f}',
                approximate_visible_duration_seconds=f'{len(frames)/fps:.3f}')


def write_reports(video_stem: str, rows: list[dict], fps: float, reports_dir: Path, *,
                  total_frames: int | None = None, elapsed_seconds: float | None = None,
                  identities: Iterable[StableIdentity] = (),
                  processing_mode: str = "full_inference") -> dict[str, Any]:
    """Write fresh reports, retaining legacy raw and stable summary filenames."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError('Report FPS must be positive')
    reports_dir.mkdir(parents=True, exist_ok=True)
    suffixes = ['tracking_log.csv', 'track_summary.csv', 'stable_identity_summary.csv',
                'stable_summary.csv', 'continuity_report.csv', 'continuity_statistics.json']
    paths = {suffix: reports_dir / f'{video_stem}_{suffix}' for suffix in suffixes}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Reports already exist; choose a fresh --output-root')
    raw: dict[int, list[int]] = defaultdict(list)
    stable: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        raw[int(row['track_id'])].append(int(row['frame_number']))
        if row.get('stable_id') is not None:
            stable[str(row['stable_id'])].append(row)
    last_processed = total_frames if total_frames is not None else max((int(r['frame_number']) for r in rows), default=0)
    state_by_id = {item.stable_id: item for item in identities}
    summaries, diagnostics = [], []
    for sid, observations in sorted(stable.items()):
        observations = sorted(observations, key=lambda row: int(row['frame_number']))
        frames = [int(row['frame_number']) for row in observations]
        raw_sequence = [int(row['track_id']) for row in observations]
        changes = sum(a != b for a, b in zip(raw_sequence, raw_sequence[1:]))
        raw_ids = list(dict.fromkeys(raw_sequence))
        state = state_by_id.get(sid)
        roll = state.roll_number if state else observations[-1].get('roll_number')
        session = state.session_id if state else observations[-1].get('session_id')
        summaries.append(dict(stable_id=sid, roll_number=roll, session_id=session,
                              raw_track_ids=';'.join(map(str, raw_ids)), **_times(frames, fps),
                              number_of_raw_id_changes=changes,
                              final_state='active' if frames[-1] == last_processed else 'inactive'))
        diagnostics.append(dict(stable_id=sid, raw_track_count=len(raw_ids), raw_id_changes=changes,
                                longest_gap_frames=max((b-a-1 for a, b in zip(frames, frames[1:])), default=0),
                                reassociation_count=sum(b-a > 1 or x != y for a, b, x, y in
                                                        zip(frames, frames[1:], raw_sequence, raw_sequence[1:])),
                                total_visible_frames=len(set(frames)), start_frame=frames[0], end_frame=frames[-1],
                                roll_number=roll))
    _write_csv(paths['tracking_log.csv'], LOG_FIELDS, rows)
    _write_csv(paths['track_summary.csv'], SUMMARY_FIELDS,
               (dict(track_id=raw_id, **_times(frames, fps)) for raw_id, frames in sorted(raw.items())))
    _write_csv(paths['stable_identity_summary.csv'], STABLE_FIELDS, summaries)
    _write_csv(paths['stable_summary.csv'], STABLE_FIELDS, summaries)
    _write_csv(paths['continuity_report.csv'], DIAGNOSTIC_FIELDS, diagnostics)
    statistics = dict(processing_mode=processing_mode, total_processed_frames=last_processed, total_raw_track_ids=len(raw),
                      total_stable_identities=len(stable),
                      raw_id_changes_per_stable_identity={r['stable_id']: r['number_of_raw_id_changes'] for r in summaries},
                      very_short_raw_tracks_lt_5_frames=sum(len(set(f)) < 5 for f in raw.values()),
                      longest_stable_trajectory_visible_frames=max((r['frames_visible'] for r in summaries), default=0),
                      average_processing_fps=last_processed / elapsed_seconds if elapsed_seconds and elapsed_seconds > 0 else None,
                      elapsed_processing_seconds=elapsed_seconds, source_fps=fps,
                      summed_person_visible_seconds=sum(len(set(f)) for f in raw.values()) / fps,
                      disclaimer=DISCLAIMER)
    paths['continuity_statistics.json'].write_text(json.dumps(statistics, indent=2) + '\n', encoding='utf-8')
    print(f"Tracking log: {paths['tracking_log.csv']}")
    print(f"Stable identity summary: {paths['stable_identity_summary.csv']}")
    return statistics
