"""Synthetic continuity checks; these are not tracking accuracy measurements."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import numpy as np
import pytest

from src.tracking.stable_identity import StableIdentityTracker
from src.tracking.identity_mapper import StableIdentityMapper
from src.tracking.person_tracker import PersonTracker


def observation(raw=1, x=10):
    return dict(track_id=raw, confidence=.9, x1=x, y1=10, x2=x+40, y2=90)


def frame(color=(30, 100, 200)):
    return np.full((120, 240, 3), color, dtype=np.uint8)


def test_short_gap_raw_change_and_expiration():
    tracker = StableIdentityTracker(max_gap_frames=3)
    first = tracker.update(frame(), [observation()])[0]['stable_id']
    tracker.update(frame(), [])
    recovered = tracker.update(frame(), [observation(8, 12)])[0]
    assert recovered['stable_id'] == first and recovered['track_id'] == 8
    for _ in range(3):
        tracker.update(frame(), [])
    assert tracker.update(frame(), [observation(8, 12)])[0]['stable_id'] != first


def test_distant_or_different_appearance_never_reconnects():
    for new_frame, box in [(frame(), observation(2, 170)), (frame((200, 20, 20)), observation(2))]:
        tracker = StableIdentityTracker()
        first = tracker.update(frame(), [observation()])[0]['stable_id']
        tracker.update(frame(), [])
        assert tracker.update(new_frame, [box])[0]['stable_id'] != first


def test_ambiguous_candidates_and_simultaneous_people():
    tracker = StableIdentityTracker()
    original = tracker.update(frame(), [observation(1, 10), observation(2, 45)])
    tracker.update(frame(), [])
    # Equally plausible past people: refuse either assignment.
    current = tracker.update(frame(), [observation(3, 27)])[0]
    assert current['stable_id'] not in {item['stable_id'] for item in original}
    current = tracker.update(frame(), [observation(3, 27), observation(4, 28)])
    assert len({item['stable_id'] for item in current}) == 2


def test_same_raw_id_jump_and_reset():
    tracker = StableIdentityTracker()
    first = tracker.update(frame(), [observation()])[0]['stable_id']
    assert tracker.update(frame(), [observation(1, 175)])[0]['stable_id'] != first
    tracker.reset()
    assert tracker.states == {} and tracker.frame_number == 0
    assert tracker.update(frame(), [observation()])[0]['stable_id'] == 'S001'


def test_overlap_does_not_supply_recovery_appearance():
    tracker = StableIdentityTracker()
    original = tracker.update(frame(), [observation()])[0]['stable_id']
    tracker.update(frame(), [])
    result = tracker.update(frame(), [observation(2), observation(3, 15)])
    assert original not in {item['stable_id'] for item in result}


def test_verified_binding_survives_raw_change_without_roll_inference():
    tracker = StableIdentityTracker()
    mapper = StableIdentityMapper('verified-session')
    stable = tracker.update(frame(), [observation(7)])[0]['stable_id']
    identity = mapper.associate_verified(stable, {'verified': True, 'roll_number': '22102030'})
    recovered = tracker.update(frame(), [observation(99)])[0]['stable_id']
    assert mapper.resolve(recovered) == identity
    assert identity.roll_number == '22102030'
    with pytest.raises(ValueError):
        mapper.associate_verified(stable, {'verified': True, 'roll_number': '22102002'})
    with pytest.raises(ValueError):
        mapper.associate_verified(2, {'verified': False, 'roll_number': '22102002'})


def test_wrapper_ages_empty_frames_and_resets_both_layers(monkeypatch):
    backend = Mock()
    backend.track.return_value = []
    backend.predictor = SimpleNamespace(trackers=[Mock()])
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=Mock(return_value=backend)))
    tracker = PersonTracker()
    for _ in range(3):
        assert tracker.track(frame()) == []
    assert tracker.stable_tracker.frame_number == 3
    assert backend.track.call_args.kwargs['conf'] == .1
    assert backend.track.call_args.kwargs['persist'] is True
    assert Path(backend.track.call_args.kwargs['tracker']).is_file()
    tracker.reset()
    backend.predictor.trackers[0].reset.assert_called_once()
    assert tracker.stable_tracker.frame_number == 0


def test_reports_keep_raw_summary_and_count_only_observed_frames(tmp_path, monkeypatch):
    import csv
    script = Path(__file__).resolve().parents[1] / 'scripts/08_test_body_tracking.py'
    spec = importlib.util.spec_from_file_location('body_tracking_runner', script)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'PROJECT_ROOT', tmp_path)
    rows = [dict(video_name='demo', frame_number=f, timestamp_seconds=(f-1)/30,
                 stable_id=1, **observation(raw)) for f, raw in [(1, 7), (4, 9)]]
    runner.write_reports('demo', rows, 30, tmp_path)
    with (tmp_path / 'demo_stable_summary.csv').open() as handle:
        summary = list(csv.DictReader(handle))
    assert summary[0]['raw_track_ids'] == '7;9'
    assert summary[0]['frames_visible'] == '2'
    assert summary[0]['approximate_visible_duration_seconds'] == '0.067'
    with (tmp_path / 'demo_track_summary.csv').open() as handle:
        assert len(list(csv.DictReader(handle))) == 2
