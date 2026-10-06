"""Synthetic contract/regression tests; no YOLO weights or network required."""
from dataclasses import replace
import csv
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from src.tracking.appearance_encoder import HistogramAppearanceEncoder, cosine_similarity
from src.tracking.person_tracker import PersonTracker
from src.tracking.reporting import write_reports
from src.tracking.stable_identity_manager import StableIdentityManager, StableIdentitySettings


def image():
    return np.full((120, 240, 3), (30, 100, 200), dtype=np.uint8)


def track(raw=7, x=10):
    return dict(track_id=raw, confidence=.91, x1=x, y1=10, x2=x+40, y2=90)


def warm(manager, raw=7, x=10):
    manager.update(image(), [track(raw, x)], 0.)
    return manager.update(image(), [track(raw, x)], .1)[0]['stable_id']


@pytest.fixture
def runner():
    location = Path(__file__).resolve().parents[1] / 'scripts/08_test_body_tracking.py'
    spec = importlib.util.spec_from_file_location('tracking_runner_test', location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stable_record_history_verified_roll_and_change_count():
    manager = StableIdentityManager()
    sid = warm(manager)
    assert sid == 'S001' and manager.states[sid].roll_number is None
    manager.associate_roll(sid, '22102030', 'fingerprint-session', verified=True)
    manager.update(image(), [], .2)
    state = manager.states[sid]
    assert not state.active and state.lost_frame_count == 1 and not state.expired
    manager.update(image(), [], .3)
    result = manager.update(image(), [track(19)], .4)[0]
    assert result['stable_id'] == sid and result['roll_number'] == '22102030'
    assert result['session_id'] == 'fingerprint-session' and result['track_id'] == 19
    manager.update(image(), [track(24)], .5)
    assert state.previous_track_ids == [7, 19]
    assert state.raw_track_ids == [7, 19, 24]
    assert state.number_of_raw_id_changes == 2
    assert state.first_seen_frame == 1 and state.last_seen_frame == 6
    assert state.first_seen_time == 0 and state.last_seen_time == .5
    assert state.frames_visible == 4 and state.longest_gap_frames == 2
    assert state.confidence == .91 and state.active
    assert manager.raw_to_stable == {24: sid}


def test_verification_required_and_conflicting_bindings_rejected():
    manager = StableIdentityManager()
    sid = warm(manager)
    with pytest.raises(ValueError, match='verified'):
        manager.associate_roll(sid, '22102030', 'session')
    manager.associate_verified(sid, {'verified': True, 'roll_number': '22102030'}, 'session')
    with pytest.raises(ValueError):
        manager.associate_roll(sid, '22102002', 'session', verified=True)
    with pytest.raises(ValueError):
        manager.associate_roll(sid, '22102030', 'another-session', verified=True)
    with pytest.raises(ValueError):
        manager.associate_roll('S999', '22102030', 'session', verified=True)
    manager.update(image(), [], .2)
    with pytest.raises(ValueError, match='currently visible'):
        manager.associate_roll(sid, '22102030', 'session', verified=True)


def test_grace_period_boundary_and_reset_clear_bindings():
    cfg = replace(StableIdentitySettings(), max_missing_frames=3, max_reassociation_gap_frames=3)
    manager = StableIdentityManager(cfg)
    sid = warm(manager)
    manager.associate_roll(sid, '22102030', 'session', verified=True)
    for t in (.2, .3, .4):
        manager.update(image(), [], t)
    assert not manager.states[sid].expired
    result = manager.update(image(), [track(7)], .5)[0]
    assert result['stable_id'] != sid and result['roll_number'] is None
    assert manager.states[sid].expired
    manager.reset()
    assert not manager.states and not manager.raw_to_stable
    assert warm(manager) == 'S001' and manager.states['S001'].roll_number is None


def test_mutual_uniqueness_rejects_two_new_tracks_for_one_lost_person():
    manager = StableIdentityManager()
    sid = warm(manager, x=27)
    manager.update(image(), [], .2)
    result = manager.update(image(), [track(19, 10), track(20, 45)], .3)
    assert sid not in {item['stable_id'] for item in result}
    assert len({item['stable_id'] for item in result}) == 2


def test_one_raw_id_has_one_stable_identity_and_duplicate_input_rejected():
    manager = StableIdentityManager()
    sid = warm(manager)
    assert manager.update(image(), [track()], .2)[0]['stable_id'] == sid
    before = manager.frame_number
    with pytest.raises(ValueError, match='Duplicate'):
        manager.update(image(), [track(), track()], .3)
    assert manager.frame_number == before
    assert manager.raw_to_stable == {7: sid}


def test_no_transfer_from_currently_visible_raw_id():
    manager = StableIdentityManager()
    sid = warm(manager)
    result = manager.update(image(), [track(7), track(19, 15)], .2)
    assert result[0]['stable_id'] == sid
    assert result[1]['stable_id'] != sid


@pytest.mark.parametrize('enabled', [True, False])
def test_unavailable_or_disabled_appearance_uses_stricter_short_gap(enabled):
    encoder = Mock()
    encoder.extract.return_value = None
    manager = StableIdentityManager(replace(StableIdentitySettings(), appearance_enabled=enabled), encoder)
    sid = warm(manager)
    assert manager.update(image(), [track(19)], .2)[0]['stable_id'] == sid
    manager.update(image(), [], .3)
    manager.update(image(), [], .4)
    assert manager.update(image(), [track(20)], .5)[0]['stable_id'] != sid
    if not enabled:
        encoder.extract.assert_not_called()


def test_appearance_history_is_bounded_normalized_and_not_single_frame():
    cfg = replace(StableIdentitySettings(), appearance_history_size=3)
    manager = StableIdentityManager(cfg)
    sid = warm(manager)
    for n in range(2, 6):
        manager.update(image(), [track()], n / 10)
    state = manager.states[sid]
    assert len(state.appearance_history) == 3
    assert np.linalg.norm(state.appearance_embedding) == pytest.approx(1.)
    assert cosine_similarity(state.appearance_embedding, state.appearance_history[-1]) == pytest.approx(1.)
    one = StableIdentityManager()
    sid = one.update(image(), [track()], 0.)[0]['stable_id']
    one.update(image(), [], .1)
    one.update(image(), [], .2)
    assert one.update(image(), [track(19)], .3)[0]['stable_id'] != sid


def test_mismatch_never_uses_geometry_fallback():
    manager = StableIdentityManager()
    sid = warm(manager)
    opposite = np.full((120, 240, 3), (200, 20, 20), dtype=np.uint8)
    assert manager.update(opposite, [track(19)], .2)[0]['stable_id'] != sid


@pytest.mark.parametrize('box', [(10, 10, 10, 20), (20, 10, 5, 20), (-100, -100, -50, -50),
                                  (300, 300, 350, 350), (0, 0, 1, 1), (float('nan'), 1, 5, 6),
                                  (0, 0, float('inf'), 10), (1, 2, 3)])
def test_invalid_appearance_crops_are_safe(box):
    assert HistogramAppearanceEncoder().extract(image(), box) is None


def test_boundary_crop_is_clamped():
    value = HistogramAppearanceEncoder().extract(image(), (-10, -10, 50, 90))
    assert value is not None and np.linalg.norm(value) == pytest.approx(1.)


def test_disable_stable_id_retains_raw_and_does_not_guess_student():
    manager = StableIdentityManager(replace(StableIdentitySettings(), enabled=False))
    result = manager.update(image(), [track(22102030)], 0.)[0]
    assert result['track_id'] == 22102030
    assert result['stable_id'] is None and result['roll_number'] is None and result['session_id'] is None
    assert not manager.states


def test_reports_fields_final_inactive_state_and_raw_id_changes(tmp_path):
    manager = StableIdentityManager()
    rows = []
    for number, raw in enumerate([7, 7, 19, 7], 1):
        obs = manager.update(image(), [track(raw)], (number-1)/30)[0]
        rows.append(dict(video_name='demo', frame_number=number, timestamp_seconds=(number-1)/30, **obs))
    manager.update(image(), [], 4/30)
    stats = write_reports('demo', rows, 30, tmp_path, total_frames=5, elapsed_seconds=1,
                          identities=manager.states.values())
    with (tmp_path / 'demo_stable_identity_summary.csv').open() as stream:
        summary = list(csv.DictReader(stream))[0]
    assert summary['stable_id'] == 'S001' and summary['raw_track_ids'] == '7;19'
    assert summary['number_of_raw_id_changes'] == '2' and summary['final_state'] == 'inactive'
    assert summary['frames_visible'] == '4' and summary['approximate_visible_duration_seconds'] == '0.133'
    with (tmp_path / 'demo_tracking_log.csv').open() as stream:
        assert {'track_id', 'stable_id', 'roll_number', 'session_id'} <= set(csv.DictReader(stream).fieldnames)
    assert stats['average_processing_fps'] == 5 and stats['total_raw_track_ids'] == 2
    with pytest.raises(FileExistsError):
        write_reports('demo', rows, 30, tmp_path)


class FakeTensor:
    def __init__(self, values):
        self.values = values
    def detach(self):
        return self
    def cpu(self):
        return self
    def tolist(self):
        return self.values


@pytest.fixture
def backend(monkeypatch):
    boxes = SimpleNamespace(xyxy=FakeTensor([[10, 10, 50, 90]]), conf=FakeTensor([.91]),
                            id=FakeTensor([7]), cls=FakeTensor([0]))
    yolo = Mock()
    yolo.track.return_value = [SimpleNamespace(boxes=boxes)]
    yolo.predictor = SimpleNamespace(trackers=[Mock()])
    constructor = Mock(return_value=yolo)
    monkeypatch.setitem(sys.modules, 'ultralytics', SimpleNamespace(YOLO=constructor))
    return yolo, constructor


def test_backend_arguments_and_cli_mode_resolution(backend, runner):
    yolo, constructor = backend
    tracker = runner.build_tracker(dict(tracker='botsort', confidence_threshold=.12, inference_size=832))
    tracker.track(image())
    kwargs = yolo.track.call_args.kwargs
    assert kwargs['persist'] is True and kwargs['classes'] == [0]
    assert kwargs['conf'] == .12 and kwargs['imgsz'] == 832
    assert Path(kwargs['tracker']).name == 'botsort_smartpresence.yaml'
    constructor.assert_called_once()
    args = runner.parse_args(['--tracker', 'botsort', '--model', 'yolo11s.pt', '--imgsz', '960', '--conf', '.1',
                              '--disable-stable-id', '--disable-appearance', '--no-preview'])
    assert args.tracker == 'botsort' and args.imgsz == 960 and args.conf == .1
    assert args.disable_stable_id and args.disable_appearance and not args.show_preview
    assert runner.parse_args(['--no-preview']).show_preview is False


def test_unavailable_backend_and_model_fail_with_clear_error(backend):
    yolo, constructor = backend
    tracker = PersonTracker(tracker_config='botsort.yaml')
    yolo.track.side_effect = ImportError('ReID unavailable')
    with pytest.raises(RuntimeError, match='botsort.yaml.*ReID unavailable'):
        tracker.track(image())
    constructor.side_effect = OSError('missing model')
    with pytest.raises(RuntimeError, match='Could not load tracking model'):
        PersonTracker()


class FakeCapture:
    def __init__(self, cv2):
        self.cv2, self.remaining, self.released = cv2, 3, False
    def isOpened(self):
        return True
    def get(self, key):
        return {self.cv2.CAP_PROP_FPS: 30, self.cv2.CAP_PROP_FRAME_WIDTH: 240,
                self.cv2.CAP_PROP_FRAME_HEIGHT: 120, self.cv2.CAP_PROP_FRAME_COUNT: 3}[key]
    def read(self):
        self.remaining -= 1
        return (True, image()) if self.remaining >= 0 else (False, None)
    def release(self):
        self.released = True


def setup_capture(monkeypatch, runner):
    captures = []
    def create(_):
        cap = FakeCapture(runner.cv2)
        captures.append(cap)
        return cap
    monkeypatch.setattr(runner.cv2, 'VideoCapture', create)
    writer = Mock()
    writer.isOpened.return_value = True
    monkeypatch.setattr(runner.cv2, 'VideoWriter', Mock(return_value=writer))
    for name in ('imshow', 'destroyAllWindows', 'imwrite'):
        monkeypatch.setattr(runner.cv2, name, Mock())
    monkeypatch.setattr(runner.cv2, 'waitKey', Mock(return_value=-1))
    return captures


def test_video_reset_once_between_independent_streams(backend, runner, monkeypatch, tmp_path):
    yolo, constructor = backend
    tracker = PersonTracker()
    captures = setup_capture(monkeypatch, runner)
    for name in ('first.mp4', 'second.mp4'):
        assert runner.process_video(Path(name), tracker, tmp_path / name, False)
        assert tracker.identity_manager.frame_number == 3
        assert len(tracker.identity_manager.states) == 1
    assert yolo.track.call_count == 6 and yolo.predictor.trackers[0].reset.call_count == 2
    constructor.assert_called_once()
    assert all(cap.released for cap in captures)


def test_webcam_preserves_identity_during_continuous_session(backend, runner, monkeypatch):
    yolo, _ = backend
    tracker = PersonTracker()
    captures = setup_capture(monkeypatch, runner)
    assert runner.process_camera(tracker, 0)
    assert yolo.track.call_count == 3
    yolo.predictor.trackers[0].reset.assert_called_once()
    assert tracker.identity_manager.states['S001'].frames_visible == 3
    assert captures[0].released


def test_camera_backend_failure_returns_false_and_releases(backend, runner, monkeypatch):
    yolo, _ = backend
    tracker = PersonTracker()
    captures = setup_capture(monkeypatch, runner)
    yolo.track.side_effect = RuntimeError('backend unavailable')
    assert not runner.process_camera(tracker, 0)
    assert captures[0].released


def test_camera_open_failure_is_safe(backend, runner, monkeypatch):
    cap = Mock()
    cap.isOpened.return_value = False
    monkeypatch.setattr(runner.cv2, 'VideoCapture', Mock(return_value=cap))
    assert not runner.process_camera(PersonTracker(), 0)


def test_known_raw_id_survives_backend_recovery_without_new_reid_evidence():
    manager = StableIdentityManager(replace(StableIdentitySettings(), appearance_enabled=False))
    sid = warm(manager)
    manager.associate_roll(sid, '22102030', 'session', verified=True)
    for n in range(2, 20):
        manager.update(image(), [], n / 10)
    result = manager.update(image(), [track(7)], 2.)[0]
    assert result['stable_id'] == sid and result['roll_number'] == '22102030'
    assert manager.states[sid].number_of_raw_id_changes == 0
    assert manager.states[sid].longest_gap_frames == 18


def test_replay_ignores_old_stable_and_student_labels_and_checks_frame_bounds(tmp_path):
    from scripts.replay_tracking_continuity import RecordedTracker
    path = tmp_path / 'raw.csv'
    row = dict(frame_number=1, stable_id='S999', roll_number='22102030', session_id='old', **track())
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    replay = RecordedTracker(path, StableIdentitySettings(), total_frames=2)
    first = replay.track(image(), 0.)[0]
    assert first['stable_id'] == 'S001' and first['roll_number'] is None
    assert replay.track(image(), .1) == []
    assert not replay.identity_manager.states['S001'].active
    with pytest.raises(ValueError, match='longer than source inference'):
        replay.track(image(), .2)
    replay.reset()
    assert replay.frame_number == 0 and not replay.identity_manager.states


def test_missing_local_model_fails_without_calling_yolo(backend, tmp_path):
    _, constructor = backend
    with pytest.raises(RuntimeError, match='unavailable locally.*No detector download'):
        PersonTracker(model_path=tmp_path / 'missing.pt', require_local_model=True)
    constructor.assert_not_called()


def test_custom_model_and_explicit_cpu_fallback_paths(backend, runner, tmp_path, monkeypatch):
    _, constructor = backend
    custom = tmp_path / 'custom.pt'
    custom.write_bytes(b'mocked weights; YOLO constructor is mocked')
    tracker = runner.build_tracker(dict(model=str(custom), require_local_model=True))
    assert tracker.model_path == str(custom)
    constructor.assert_called_with(str(custom))
    monkeypatch.setattr(runner, 'PROJECT_ROOT', tmp_path)
    cached = tmp_path / 'models/tracking/yolo11n.pt'
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b'mocked weights')
    fallback = runner.build_tracker(dict(model='yolo11n.pt', require_local_model=True))
    assert fallback.model_path == str(cached)


def test_explicit_diagnostic_limit_keeps_consecutive_frames(backend, runner, monkeypatch, tmp_path):
    yolo, _ = backend
    setup_capture(monkeypatch, runner)
    tracker = PersonTracker()
    assert runner.process_video(Path('prefix.mp4'), tracker, tmp_path, False, max_frames=2)
    assert yolo.track.call_count == 2 and tracker.identity_manager.frame_number == 2
    args = runner.parse_args(['--max-frames', '300', '--require-local-model'])
    assert args.max_frames == 300 and args.require_local_model


def test_cpu_threads_use_public_prediction_callback(backend, monkeypatch):
    yolo, _ = backend
    setter = Mock()
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=setter))
    PersonTracker(cpu_threads=1)
    event, callback = yolo.add_callback.call_args.args
    assert event == 'on_predict_start'
    callback(SimpleNamespace(device=SimpleNamespace(type='cpu')))
    setter.assert_called_once_with(1)
    callback(SimpleNamespace(device=SimpleNamespace(type='cuda')))
    setter.assert_called_once()
    with pytest.raises(ValueError, match='cpu_threads'):
        PersonTracker(cpu_threads=0)
