# Tracking implementation report

The final architecture is **raw tracker ID → stable session visual identity → verified roll number**. A raw ID change can preserve a stable/student identity only after conservative reassociation. No ID is forced to remain constant.

## Files and reasons

The table covers the cumulative tracking changes relative to the repository HEAD, including the earlier ByteTrack/configuration work in this session. No commit or push was made.

| Created source/config/document/test file | Purpose |
|---|---|
| `config/bytetrack_smartpresence.yaml` | Explicit classroom ByteTrack recovery thresholds and buffer |
| `config/botsort_smartpresence.yaml` | Optional fixed-camera BoT-SORT with native-feature ReID using verified supported keys |
| `src/tracking/appearance_encoder.py` | Optional clamped torso descriptor interface, normalization, cosine comparison, history aggregation |
| `src/tracking/stable_identity_manager.py` | Configurable S001-style identities, one-to-one recovery, lost-state history, verified roll/session integration |
| `src/tracking/stable_identity.py` | Compatibility wrapper for the initial stable-tracker import; one manager implementation |
| `src/tracking/reporting.py` | Separate raw/stable/diagnostic reporting, descriptive statistics, overwrite refusal |
| `scripts/replay_tracking_continuity.py` | Replay saved raw observations through the current manager without detector inference; retains final-check provenance |
| `tests/test_tracking_continuity.py` | Initial continuity/report regression tests |
| `tests/test_stable_identity_manager.py` | Full synthetic state/history/appearance/CLI/video/webcam/error-path coverage |
| `docs/tracking_identity_continuity.md` | Audit, design, identity semantics, thresholds, API contract and scientific limits |
| `docs/tracking_implementation_report.md` | Exact deliverable inventory, commands, validation results and experiment interpretation |

| Modified existing file | Purpose |
|---|---|
| `config/settings.yaml` | YOLO11s/960/0.10 defaults and explicit `STABLE_IDENTITY` configuration |
| `src/tracking/person_tracker.py` | Persistent configurable backend, stable-manager integration, metadata, annotation and actionable errors |
| `src/tracking/identity_mapper.py` | Separate verified stable-student mapping; supports S001 and legacy integer stable IDs; original raw mapper unchanged |
| `scripts/08_test_body_tracking.py` | Tracker/model/size/confidence/disable CLI options, stream timestamps, safe lifecycle, fresh output directories and run configuration |
| `requirements.txt` | Pin runtime-tested `ultralytics==8.4.138` |
| `src/utils/repository_validation.py` | Update only stale tracking assertions for the tuned backends and YOLO11s |
| `tests/test_research.py` | Update only old default tracker/model expectations |
| `README.md` | Tracking architecture, commands, outputs, optional ReID and limitations |

Downloaded artifact: `models/tracking/yolo11s.pt` (official pretrained YOLO11s detector weights). Original YOLO11n weights and face classifier artifacts were preserved. No new learned SmartPresence appearance model or dependency stack was added. The user-supplied untracked `dataset/test_videos/test_video_02.mp4` is unrelated pre-existing work and was not changed.

Generated validation artifacts live in fresh `outputs/tracking/<mode>/validation_*/` directories (exact paths below). Completed runs have a console log, run configuration, annotated video, available sample frames, raw log/summary, stable summaries, continuity diagnostics and statistics. Interrupted attempts retain only their partial files and are excluded from completed-run statistics. The no-ReID comparison additionally saves the exact `botsort_no_reid.yaml` used. The final validation manifest lists every generated file and its hash.

## Final parameters

| Parameter | ByteTrack | BoT-SORT |
|---|---|---|
| tracker_type | bytetrack | botsort |
| track_high_thresh | 0.45 | 0.45 |
| track_low_thresh | 0.10 | 0.10 |
| new_track_thresh | 0.55 | 0.55 |
| track_buffer | 120 | 120 |
| match_thresh | 0.80 | 0.80 |
| fuse_score | true | true |
| gmc_method | — | none |
| proximity_thresh | — | 0.50 |
| appearance_thresh | — | 0.85 |
| with_reid | — | true |
| model (backend ReID) | — | auto |

Default detector: **models/tracking/yolo11s.pt**, person class **0**, confidence **0.10**, image size **960**. Preview defaults true; sample-frame setting remains 3, with the established first/middle/final snapshot selection. Every consecutive frame is processed. Known raw IDs retain continuity through the grace period after sanity checks; stricter reassociation gates apply to changed raw IDs.

The stable layer uses 120-update retention/recovery, minimum IoU 0.30, center distance ≤0.35 previous-box diagonals, size ratios 0.65–1.55, bounded five-frame motion extrapolation, cosine appearance similarity ≥0.90, history up to 20 descriptors with at least two historical samples, and mutual uniqueness. Similarity failures/ambiguity split identities. Without sufficient appearance, only a two-update geometry fallback (IoU ≥0.70, distance ≤0.15 diagonals) can recover. Full settings and behavior are in [the design](tracking_identity_continuity.md).

BoT-SORT **actually has backend ReID enabled** in the supplied config (`with_reid: true`, `model: auto`) and was runtime-tested with YOLO11s. SmartPresence's separate appearance descriptor is an optional normalized HSV histogram, not learned ReID. No private Ultralytics embedding internals are read or patched by SmartPresence.

## Commands and locations

Run in an environment with `requirements.txt` installed and a working Python interpreter, from the repository root:

```bash
# Existing usage remains valid; tuned ByteTrack is default:
python scripts/08_test_body_tracking.py --no-preview
# Explicit tuned ByteTrack:
python scripts/08_test_body_tracking.py --tracker bytetrack --no-preview
# BoT-SORT + native-feature backend ReID:
python scripts/08_test_body_tracking.py --tracker botsort --no-preview
# Webcam (one persistent session):
python scripts/08_test_body_tracking.py --camera --tracker botsort
# Explicit detector controls:
python scripts/08_test_body_tracking.py --tracker bytetrack --model models/tracking/yolo11s.pt --imgsz 960 --conf 0.10 --no-preview
# Independent descriptor/raw-only controls:
python scripts/08_test_body_tracking.py --disable-appearance --no-preview
python scripts/08_test_body_tracking.py --disable-stable-id --no-preview
```

Input videos: **`test_data/videos/body_tracking/`**. Default output root: **`outputs/tracking/runs/<UTC-time>_<suffix>/`**, printed by the runner. Supply a fresh `--output-root` to choose a fixed location. Under that root:

- Videos: `videos/<video>_tracked.mp4`.
- Per-frame logs: `reports/<video>_tracking_log.csv`.
- Stable identity summaries: `reports/<video>_stable_identity_summary.csv`.
- Raw summaries, stable-summary compatibility alias, continuity CSV/JSON, and snapshots retain the names documented in the design.

`--disable-appearance` affects the SmartPresence encoder, not BoT-SORT's backend ReID flag. The no-ReID experiment used a saved custom config with `with_reid: false` via the same configurable `--tracker` path API. Webcam capture was covered with mocks; no physical webcam was opened during validation.

## Automated checks

The tests use mocks/synthetic data and require no weights download. They cover all requested behaviors, including persistence/class/config forwarding, grace/expiry, one-to-one matching, roll preservation, history/change counts, CSVs, disabled/unavailable appearance, invalid crops, backend failures, video resets and continuous webcam sessions. Existing tests also cover face embedding/classifier contracts, inference normalization, fingerprint lookup/session creation, participant metadata, and attendance boundaries. They do not retrain or run the expensive face pipeline.

Required commands:

```bash
pytest -q
python -m compileall -q src scripts
python scripts/validate_research_repository.py
```

The actual lightweight validation interpreter was `/tmp/smartpresence-tracking-tests/bin/python`; the repository virtualenv has a dangling Python link. Real YOLO runs loaded the existing site packages using `PYTHONPATH=.venv/lib/python3.11/site-packages`. This leaves the repository virtualenv unchanged and does not claim a clean-install test of the entire dependency stack.

## Scientific limits

Fingerprint-to-visible-person **automatic association remains incomplete/manual**. The caller selects a currently visible stable identity after existing fingerprint/session verification, then calls `tracker.identity_manager.associate_verified(stable_id, verification, session_id)`. Its roll stays attached through accepted reassociation; no raw ID is converted to a roll. Bindings are session-local and reset at a new stream; persistence/unified attendance were not added.

No guarantee is made after full occlusion, long disappearance, leaving the field of view, severe detector failure, similar-looking people crossing, pose/lighting changes, tiny boxes, or long-interval re-entry. Stable counts can exceed raw counts when sanity checks split a raw trajectory. Fewer fragments alone do not prove correct identity assignment. The baseline uses different detector size/resolution and is not a controlled ablation of the stable layer alone.

**Tracking continuity statistics are descriptive and do not represent IDF1, HOTA, MOTA, or identity accuracy.** These outputs also cannot establish fingerprint recognition performance or end-to-end attendance accuracy. Formal claims require ground-truth identity trajectories and appropriate experimental controls.

## Final-manager replay provenance

The first full tuned runs exposed unnecessary splits when the backend recovered the same raw ID after a detection gap. The manager was corrected to keep a known raw identity within its grace period after position/size/appearance sanity checks, while keeping strict gates for reassociation to a different raw ID. A dedicated regression protects this behavior.

Earlier full-inference outputs were preserved. Tuned ByteTrack's complete raw observations were replayed through the final manager into a separate `_final` directory:

```bash
python scripts/replay_tracking_continuity.py --video test_data/videos/body_tracking/test_video_02.mp4 --source-run outputs/tracking/tuned_bytetrack/validation_20261002 --output-root outputs/tracking/tuned_bytetrack/validation_20261002_final
```

The adapter consumes only raw IDs/confidences/boxes; old stable and roll fields are discarded. Replay does not rerun the detector, and replay FPS is **not inference FPS**. The October 2 no-ReID BoT-SORT full run was interrupted after progress reached frame 1,600 and produced no completed CSV reports. Its partial artifacts are preserved and excluded from statistics. Its planned full ReID run never started. The October 4 BoT-SORT checks use the final manager with full detector inference over the same **300-frame consecutive prefix**, once with backend ReID and once without. These bounded checks are not full-video validations.

## Offline and CPU model selection

YOLO11s at 960 remains the recommended default. Custom model paths and bare cached names are supported. No model is silently substituted. For a local-only detector load or explicit CPU fallback:

```bash
python scripts/08_test_body_tracking.py --tracker bytetrack --model models/tracking/yolo11s.pt --require-local-model --no-preview
python scripts/08_test_body_tracking.py --tracker bytetrack --model yolo11n.pt --imgsz 640 --require-local-model --no-preview
```

`--require-local-model` checks detector weights before YOLO construction. It does not promise to disable networking in every dependency or in backend fallbacks for other detector architectures. Without it, Ultralytics may download recognized missing weights; original loading errors are retained and accompanied by an explicit fallback suggestion. All unit tests mock YOLO; none require real weights or download a detector.

The additional `--max-frames 300` option provides a bounded diagnostic while preserving every consecutive frame within the prefix. The default remains the whole video. Console output and `run_config.json` identify the limit.

For constrained CPUs, `--cpu-threads 1` is available through a public Ultralytics callback and the PyTorch thread API. No tracker internals are patched. The completed October 4 diagnostics use this flag; earlier slow default-thread prefix attempts were stopped and preserved without completed reports.

## Exact validation results (completed October 4, 2026)

**90 tests passed**; syntax compilation passed. Repository validation: **33 passes, 0 failures, 6 pre-existing warnings**. The warnings concern metadata provenance differences, an empty historical metrics placeholder, licensing, and already tracked processed data. Evidence is saved under `outputs/tracking/validation_20261004/`.

All 38 protected source/metadata/attendance-log/historical tracking files remain byte-identical to the saved pre-experiment hashes. Earlier checks also confirmed tracked model artifacts match HEAD. The existing face pipelines were not retrained or run end-to-end; their existing contract tests and metadata/artifact checks passed. No physical webcam was opened; continuous webcam state was tested with mocks.

Input: `test_data/videos/body_tracking/test_video_02.mp4`, 2,225 frames at 30 FPS, 848 × 478, approximately 74.17 seconds. Baseline/tuned runs cover the full input. The two completed BoT-SORT checks cover the same first 300 frames (10 seconds), **not the full video**. Comparing counts across these different lengths is not a controlled benchmark.

| Mode | Processed frames | Raw IDs | Final stable IDs | Raw changes in stable records | Raw tracks <5 frames | Longest stable visible frames | Measured inference-run FPS | Summed person-visible seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2225 | 26 | disabled | — | 0 | — | 15.94 | 357.43 |
| tuned_bytetrack | 2225 | 8 | 8 | 0 | 0 | 2225 | 3.80 | 375.77 |
| stable_botsort | 300 | 5 | 5 | 0 | 0 | 300 | 2.13 | 43.33 |
| stable_botsort_reid | 300 | 5 | 5 | 0 | 0 | 300 | 2.14 | 43.33 |

FPS includes inference, decoding, annotation and encoding. It excludes model construction but includes first prediction warmup. The completed October 4 runs explicitly use `--cpu-threads 1` and ran concurrently. Earlier default-thread attempts were stopped, retained as partial artifacts, and excluded from statistics. CPU conditions differ across dates, so timings are not a controlled speed comparison. Tuned stable counts use final-manager replay; its replay-only FPS is excluded from the table. The baseline used YOLO11n/640/confidence 0.40 and packaged ByteTrack, whereas tuned modes use YOLO11s/960/confidence 0.10; the comparison cannot isolate the effect of stable matching alone.

Final result roots (each has `videos/test_video_02_tracked.mp4`, `reports/test_video_02_tracking_log.csv`, and `reports/test_video_02_stable_identity_summary.csv`):

- `outputs/tracking/baseline/validation_20261002`
- `outputs/tracking/tuned_bytetrack/validation_20261002_final`
- `outputs/tracking/stable_botsort/validation_20261004_prefix300_cpu1`
- `outputs/tracking/stable_botsort_reid/validation_20261004_prefix300_cpu1`

The tuned first-pass stable layer produced 28 identities for eight raw IDs; the preserved final-manager replay has eight stable identities for eight raw IDs. Its real-video sequence accepted no cross-raw reassociations. Synthetic tests demonstrate strong reassociation with preserved roll/session bindings, but this clip does not establish correctness or effectiveness of that behavior. No student rolls were inferred or assigned in any video run.

`outputs/tracking/validation_20261004/comparison.json` contains complete per-stable raw-change dictionaries and all run statistics. `artifact_manifest.json` lists generated files, hashes, source/model/input fingerprints and runtime versions. The interrupted October 2 BoT-SORT artifacts are included in the inventory but excluded from completed-run statistics.

Runtime: Python 3.11.16, Ultralytics 8.4.138, PyTorch 2.14.0+cpu, OpenCV 5.0.0, NumPy 2.0.2, CPU-only. Test/runtime environments use the documented temporary interpreter because the repository virtualenv interpreter link is broken.

**Tracking continuity statistics are descriptive and do not represent IDF1, HOTA, MOTA, or identity accuracy.** Fingerprint-to-visible-person selection remains manual, long/ambiguous gaps can break continuity, and full-video BoT-SORT evaluation plus trajectory ground truth remain future validation work. No commit or push was made.
