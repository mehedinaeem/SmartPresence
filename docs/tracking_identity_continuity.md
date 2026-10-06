# Tracking audit, design, and reproducibility

## Audit findings

The original YOLO runner already constructed one model, processed every consecutive frame with `persist=True`, and reset once before each independent video/webcam session. There was no periodic reset or model recreation inside the loop. The confidence floor of 0.40 discarded detections ByteTrack could use for its second association stage (low threshold 0.10). Its packaged default buffer was 30 frames. The tuned detector passes scores down to 0.10 and uses a 120-frame buffer.

Raw IDs appeared in `PersonTrack`, annotations, per-frame CSVs, and raw summaries. `IdentityMapper` accepted an explicit roll and a raw ID but had no production callers. No numerical inference from tracker IDs to roll numbers was found. The fingerprint scripts verify the static lookup and call `start_session(roll, fingerprint_id)`; they do not identify a visible person. Presence logging takes an explicit roll; attendance aggregates by session and roll. The HOG/centroid scripts 09/10 remain legacy baselines.

FaceNet-SVM, all preprocessing/training/inference assets, fingerprint lookup, 75% attendance threshold, and 3-second recorded / 180-second intended CCTV observation intervals are unchanged. Those observation intervals must not be applied to body tracking: the trackers need consecutive frames.

## Three separate identities

```text
YOLO person detection
  -> ByteTrack or BoT-SORT
  -> track_id (temporary raw trajectory)
  -> StableIdentityManager
  -> stable_id (S001, S002, ... within this stream)
  -> explicit verified fingerprint/session association
  -> roll_number (permanent student identity)
```

A stable record keeps current/previous raw IDs, first/last frame and time, box, confidence, lost-frame count, active/expired status, visibility count, appearance history, motion estimate, change/reassociation counts, and optional roll/session. A missing observation marks the record inactive. It remains eligible for the grace period; expired records are retained for diagnostics but cannot recover, and their appearance history is released. Missing frames are never invented as presence.

`reset()` clears raw and stable state and all roll bindings at independent stream boundaries. Stable IDs restart at `S001` and must not be joined across streams. Raw IDs remain in every output, even if the stable layer is disabled. Disabled mode returns `None` for stable/student identity.

## Conservative reassociation

`STABLE_IDENTITY` in `config/settings.yaml` is authoritative. The old `TRACKING.stable_max_gap_frames` key remains a fallback when the new section omits retention settings.

| Gate | Default |
|---|---:|
| Missing-frame retention / maximum reassociation interval | 120 / 120 updates |
| Minimum last-box IoU | 0.30 |
| Maximum center distance / previous box diagonal | 0.35 |
| Width and height ratios | 0.65–1.55 |
| Current-person overlap threshold that blocks recovery | IoU > 0.20 |
| Appearance cosine similarity | at least 0.90 |
| Appearance history / minimum prior samples | 20 / 2 |
| Minimum confidence for appearance extraction | 0.45 |
| Motion prediction horizon | up to 5 updates |
| Geometry-only maximum interval / minimum IoU / maximum distance ratio | 2 / 0.70 / 0.15 |

A currently known raw ID retains its stable ID after size, position, and available appearance sanity checks. The same raw ID returning within retention also uses these sanity checks; stricter reassociation gates apply when the raw ID changes. Physically implausible jumps or strong appearance changes can deliberately split even an unchanged raw ID. This guards against some backend swaps but cannot detect all of them.

New raw IDs compete only for recent lost candidates, excluding raw IDs still visible in the current frame. Recovery applies temporal, last-position, size, overlap, and bounded motion-prediction gates. Appearance is compared against the normalized mean of recent L2-normalized descriptors. A match must be unique for the new observation and for the old candidate; ambiguity creates a new stable ID. It never uses an unthresholded nearest-neighbor rule.

When appearance is disabled, invalid, or has fewer than two historical samples, only the stricter two-update geometry fallback is available. A contradictory descriptor cannot be bypassed by that fallback. Overlapping current boxes do not provide new appearance evidence or recover lost identities. These values are conservative starting parameters, not tuned ground-truth performance claims. Intervals count processed frames, not seconds; 120 updates are roughly four seconds at 30 FPS.

## Two different appearance mechanisms

**BoT-SORT backend ReID:** `config/botsort_smartpresence.yaml` sets `with_reid: true` and `model: auto`. Ultralytics internally uses YOLO11 detector features with these tested weights. This mode was exercised on the existing video. Native feature matching is enabled; it is not a separately trained classroom person-ReID model. Other detector architectures may trigger an Ultralytics fallback model download. All config keys were checked against the installed default before creating the custom file. A fixed camera uses supported `gmc_method: none`. [Official tracker documentation](https://docs.ultralytics.com/modes/track/).

**SmartPresence stable appearance:** the tested public `Results` interface does not offer a documented, stable per-person embedding contract. SmartPresence therefore does not inspect private feature hooks or tracker state. `AppearanceEncoder.extract(frame, bbox)` is replaceable; the default implementation uses central torso HSV histograms with clamped crops, invalid-crop handling, L2 normalization, cosine similarity, and a bounded appearance history. It is a lightweight clothing-color heuristic, not learned person ReID, and uses only existing OpenCV/NumPy dependencies. It never invokes FaceNet on body crops.

`--disable-appearance` disables this SmartPresence descriptor only; it does not change BoT-SORT's YAML. To disable backend ReID, make a separate YAML copy with `with_reid: false` and pass its path through `--tracker`. ByteTrack never uses a backend ReID encoder.

## Verified student binding

After a caller verifies a fingerprint and explicitly selects the correct currently visible stable ID:

```python
verification = verify_fingerprint("FP030")  # existing simulated lookup
# session_id comes from the existing verified session workflow.
# selected_stable_id comes from an explicit human selection, never from FP030 or a raw ID.
tracker.identity_manager.associate_verified(selected_stable_id, verification, session_id)
```

The equivalent explicit assertion API is `associate_roll("S003", "22102030", session_id, verified=True)`. Without `verified=True`, it refuses the binding. These APIs enforce verified-result flags, valid rolls, currently visible stable IDs, and nonconflicting bindings. They are not authentication. A student already bound elsewhere is not silently merged or rebound. After a conservative reassociation, the same record retains the roll and session, e.g. `T7 -> S003 -> 22102030`, then `T19 -> S003 -> 22102030`.

Automatic fingerprint-to-visible-person association remains **incomplete/manual**. There is no validated automatic selection rule. The original raw `IdentityMapper` remains for legacy callers; new tracking uses `StableIdentityMapper` through the manager. No automatic attendance connection or persistent student binding was added.

## Backend and environment reproducibility

`requirements.txt` pins **ultralytics==8.4.138**, the installed version inspected and runtime-tested with YOLO11s, ByteTrack, BoT-SORT without ReID, and BoT-SORT with native-feature ReID. No additional learned appearance model/dependency stack was introduced. `--model` accepts custom paths and cached bare names, including the explicit YOLO11n CPU fallback. `--require-local-model` refuses missing detector weights before calling YOLO; it is a local detector-weight check, not a network firewall for all dependencies or optional backend models. Missing weights outside local-only mode may be downloaded by Ultralytics; failures retain the original exception and never silently select another model. YOLO11s weights were downloaded to `models/tracking/yolo11s.pt`; the original YOLO11n weights remain intact.

The repository `.venv/bin/python` points to a missing interpreter. It was not repaired or overwritten. Lightweight tests use `/tmp/smartpresence-tracking-tests/bin/python` with `requirements-test.txt`. Real inference uses that working Python 3.11 interpreter with `PYTHONPATH=.venv/lib/python3.11/site-packages` to load the existing Ultralytics/PyTorch installation. Thus successful inference validates those installed packages, not a clean installation of the whole face-recognition stack. Runtime is CPU-only; exact versions and hashes are recorded with the validation results.

Every new CLI run saves effective parameters, dependency versions, source/model/input hashes in `run_config.json` and uses a fresh `outputs/tracking/runs/<UTC-time>_<suffix>/` directory unless `--output-root` is supplied. Existing reports/videos are refused rather than overwritten. The prior raw-summary format and `_stable_summary.csv` alias remain available.

## Outputs and measurement limits

For output root `R` and input stem `V`:

- `R/videos/V_tracked.mp4`: labels `S003 | T19 | 91.0%` or `22102030 | S003 | 91.0%`.
- `R/frames/`: available first, middle, final tracked snapshots.
- `R/reports/V_tracking_log.csv`: raw/stable/roll/session IDs and observed box/confidence for each frame.
- `R/reports/V_track_summary.csv`: original raw-ID summary.
- `R/reports/V_stable_identity_summary.csv`: stable IDs, optional roll/session, raw history, visibility, raw-change count, final active/inactive state.
- `R/reports/V_stable_summary.csv`: compatibility alias of the stable summary.
- `R/reports/V_continuity_report.csv`: gaps, reassociations, raw changes, visibility counts.
- `R/reports/V_continuity_statistics.json`: frame/raw/stable counts, raw changes by identity, tracks shorter than five visible frames, longest stable visibility, elapsed time, average processing FPS, and summed person-visible duration.

Durations are observed frames / source FPS, not first-to-last elapsed spans. Summed person-visible seconds can exceed video duration when several people are visible. FPS includes frame decoding, inference, annotation, encoding, and selected snapshot writes; it is not a hardware-independent model benchmark. Webcam mode is continuous preview (or headless with `--no-preview`), with no video CSV export.

There is no guarantee of immutable raw IDs or stable continuity after full occlusion, long disappearance, leaving the field of view, strong detection failure, similar-looking people crossing, severe pose changes, major lighting changes, very small boxes, or re-entry after a long interval. Appearance is fallible. Stable IDs can split and can still be wrong; conservative gating cannot prove person identity. Creating `S004` is preferable to forcing an uncertain match to `S003`.

**Tracking continuity statistics are descriptive and do not represent IDF1, HOTA, MOTA, or identity accuracy.** Ground-truth trajectory annotations are required to claim improved identity correctness. See [the implementation and validation report](tracking_implementation_report.md) for exact runs and results.

`--max-frames N` optionally bounds a recorded diagnostic to the first N consecutive frames; default runs still process the entire video. Frame limits are saved in `run_config.json` and clearly reported in the console. Do not compare fragment counts across different video lengths as a controlled benchmark.

`--cpu-threads N` optionally sets PyTorch intra-op CPU threads through Ultralytics' supported public `on_predict_start` callback. This controls CPU oversubscription and does not alter detector/matching parameters. The setting is recorded in each run manifest; GPU execution is unaffected.
