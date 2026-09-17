# Research cleanup report

## Issues addressed

Local training used a plain linear SVM instead of the imported normalized RBF pipeline. FaceNet crop embedding extraction lacked explicit input normalization. Training/evaluation could overwrite imported artifacts and baseline results. The model-check command expected 40 face classes instead of 36. Two numbered image scripts referenced removed APIs. Session-only and legacy tracking entry points lacked sufficiently explicit descriptions. Dataset histories, software environment, licensing, and tracked-image release concerns required documentation.

## Changes

Future training uses L2 Normalizer + RBF SVC (C=10, gamma=scale, balanced classes, probability enabled, seed 42, test split 0.20). Future crop extraction explicitly uses Facenet normalization, skip detection and 128-D validation. Existing inference normalization and thresholds are unchanged. No real model was trained during this cleanup.

Extraction/training/evaluation reserve new run directories, record hashes and source identities, and refuse existing run IDs. Training validates embedding provenance and labels, records exact split indices and saves its own model artifacts; evaluation selects a training run explicitly and writes a separate run. No historical results were reconstructed. The 5,737-crop preprocessing snapshot and 6,901-embedding imported run remain distinct. Exact historical input normalization/environment equivalence is not established.

New read-only validation/status commands, documented compatibility aliases, optional supplied-score threshold analysis, lightweight CI and tests support future work. Legacy HOG/centroid code is retained and labeled; YOLO11n/ByteTrack still processes consecutive frames. Identity mapping documentation requires prior verification and forbids inferring rolls from track numbers.

Privacy ignore rules protect future processed/private images and run pickle files. They do not remove the 6,901 processed images already tracked or rewrite history. No license was selected. Citation authorship remains an explicit placeholder for the owner to confirm. No publication or ethics approval was invented.

## Created files

- `.github/workflows/tests.yml`
- `CITATION.cff`
- `DATA_PRIVACY.md`
- `LICENSE_RECOMMENDATION.md`
- `docs/CLEANUP_REPORT.md`
- `docs/JOURNAL_REPRODUCIBILITY_CHECKLIST.md`
- `environment_versions.json`
- `experiments/README.md`
- `models/README.md`
- `pytest.ini`
- `requirements-test.txt`
- `scripts/09_calculate_attendance.py`
- `scripts/10_run_covered_face_tracking.py`
- `scripts/11_start_fingerprint_session.py`
- `scripts/12_run_full_system.py`
- `scripts/evaluate_unknown_threshold.py`
- `scripts/research_status.py`
- `scripts/validate_research_repository.py`
- `src/utils/repository_validation.py`
- `src/utils/research.py`
- `tests/test_research.py`

## Modified files

- `.gitignore`
- `README.md`
- `config/settings.yaml`
- `scripts/02_extract_embeddings.py`
- `scripts/03_train_model.py`
- `scripts/04_evaluate_model.py`
- `scripts/05_test_single_image.py`
- `scripts/06_test_group_image.py`
- `scripts/10_run_full_system.py`
- `scripts/check_model.py`
- `src/recognition/embedding_extractor.py`
- `src/recognition/facenet_embedder.py`
- `src/recognition/svm_evaluator.py`
- `src/recognition/svm_trainer.py`
- `src/tracking/covered_face_tracker.py`
- `src/tracking/identity_mapper.py`
- `src/tracking/person_detector.py`
- `src/tracking/tracker.py`

## Intentionally unchanged

SHA-256 checks of all 15,363 pre-existing files under dataset/, models/, experiments/, outputs/, attendance_logs/ and test_data/ found no changes. This includes saved classifier/encoder/YOLO binaries, model metadata, accuracy/F1 values, experiment CSVs, participant records and images. New README files in models/ and experiments/ are additions only. The saved face recognizer implementation, YOLO/ByteTrack implementation, fingerprint lookup, attendance algorithms, 75% threshold and 3/180-second intervals are unchanged. requirements.txt retains scikit-learn==1.6.1; no arbitrary runtime lockfile was created.

## Verification

- 49 pytest tests passed under Python 3.11.16, including all existing tests.
- Added coverage: configuration parser and saved metadata, pipeline construction, mocked FaceNet normalization/dimension, no double inference normalization, attendance percentages, temporary track identity, per-frame tracking, deterministic dataset hashing, immutable run directories, training provenance/split and output isolation, evaluation output isolation, validation-policy drift, missing calibration inputs and numbered CLI help.
- Training/vision dependencies are mocked where appropriate; synthetic unit fixtures and their metrics exist only in temporary test directories, not research outputs.
- `python -m compileall -q src scripts` passed.
- `git diff --check` passed.
- Saved classifier inspection passed: 36 expected uncovered classes, 128-D input, Normalizer/SVC pipeline, no load compatibility warnings.
- The cleanup test environment is recorded in environment_versions.json; it is not the original Colab environment or a full vision-runtime validation. The original project .venv points to a removed Snap Python; tests used a separate temporary Python 3.11 environment.

Read-only validator output:

```text
SmartPresence Research Repository Validation

[PASS] 40 participant records
[PASS] 36 uncovered participants
[PASS] 4 covered participants
[PASS] covered rolls match expected list
[PASS] continuous roll range 22102001–22102040
[PASS] attendance threshold = 75%
[PASS] experiment sampling = 3 s
[PASS] realtime sampling = 180 s
[PASS] FaceNet dimension = 128
[PASS] explicit FaceNet input normalization
[PASS] future training = L2 Normalizer + configured RBF SVM
[PASS] ByteTrack / person class 0
[PASS] YOLO11n model configured
[PASS] YOLO model file available
[PASS] prototype unknown threshold = 0.55 (not calibrated)
[PASS] classifier available
[PASS] label_encoder available
[PASS] model_info available
[PASS] students_metadata available
[PASS] dataset_summary available
[PASS] face_visibility available
[PASS] fingerprint_database available
[PASS] metadata/preprocessing_summary.csv available
[PASS] metadata/student_folder_mapping.csv available
[PASS] metadata/processed_images.csv available
[PASS] metadata/rejected_images.csv available
[PASS] 40 unique static fingerprint mappings validated against students.csv
[PASS] saved metadata dimension = 128
[PASS] saved metadata describes 36 face classes
[PASS] saved SVM metadata matches future training
[PASS] saved split and seed match configuration
[PASS] saved embedding and split counts agree
[WARNING] local preprocessing snapshot (5737) differs from imported Colab embeddings (6901); these are distinct histories
[WARNING] current processed files (6901) differ from preprocessing metadata (5737); freeze/reconcile provenance before training
[WARNING] imported metadata does not explicitly record embedding normalization; future runs record it, historical equivalence is unverified
[WARNING] experiments/exp_001_baseline/metrics.json: empty historical placeholder, not a result
[PASS] existing preprocessing report available
[WARNING] no LICENSE assigned; owner must choose code and data terms before release
[WARNING] 6901 processed files already tracked by Git; .gitignore does not remove them or their history

Metadata and existence checks only; model binaries were not loaded.
```

## Commands

In a working Python 3.11 environment:

```bash
python -m pip install -r requirements.txt
python scripts/validate_research_repository.py
pytest -q
python -m compileall -q src scripts
python scripts/check_model.py
python scripts/research_status.py
python scripts/test_static_fingerprint.py FP030
```

Only when intentionally starting a new study run:

```bash
python scripts/02_extract_embeddings.py --run-id embeddings_example
python scripts/03_train_model.py --embeddings experiments/embeddings_example/embeddings.pkl --run-id training_example
python scripts/04_evaluate_model.py --training-run experiments/training_example --run-id evaluation_example
```

Existing inference/tracking demos retain their output locations and can overwrite demo reports; archive results before rerunning. Preprocessing also retains its explicitly invoked replacement behavior. The immutability changes apply to future extraction/training/evaluation runs.

## Manual steps before submission

Resolve the dataset-history mismatch; freeze the actual image manifest, acquisition provenance and split. Confirm independent validation data before calibrating the prototype unknown threshold. Review participant consent, institutional requirements and already-tracked image history before public release. Select code/data/model licensing terms, replace citation authorship placeholders, record actual hardware/software versions and commit, and match paper numbers/figures to preserved run artifacts. Disclose simulated fingerprint lookup and incomplete covered-track/session integration. Follow JOURNAL_REPRODUCIBILITY_CHECKLIST.md.
