# SmartPresence

SmartPresence is a reproducible hybrid attendance research system. Fingerprint verification establishes permanent student identity and opens/closes sessions. Uncovered students are observed through sampled CCTV frames, face detection, 160×160 crops, 128-D FaceNet embeddings, and an SVM labeled by roll number. Covered students use fingerprint-to-track association and body tracking. Attendance is awarded when observed presence is at least the configured 75% threshold.

## Architecture

Recorded experiment video → frame sampling every 3 seconds → face detection/cropping → FaceNet → SVM → frame-wise presence → attendance decision.

Realtime CCTV → frame sampling every 180 seconds → recognition or covered-person tracking → presence logging → attendance decision.

Fingerprint templates never enter the vision pipeline. A `track_id` is temporary and session-specific; `roll_number` is always the permanent identity.

## Structure

- `config/`: paths and experiment parameters
- `dataset/`: raw, processed, rejected, tests, metadata, and preserved legacy data
- `src/`: reusable preprocessing, recognition, fingerprint, tracking, attendance, and video code
- `scripts/`: numbered experiment entry points
- `models/`: embeddings, classifiers, tracking assets, and model metadata
- `experiments/`: immutable experiment configurations and results
- `outputs/`: images, figures, reports, tables, and annotated videos
- `attendance_logs/`: fingerprint, session, presence, tracking, and final attendance schemas
- `notebooks/`: analysis-only notebooks (to be populated as results become available)
- `tests/`: fast configuration and identity tests

See `outputs/reports/migration_report.md` for the complete legacy-data audit and assumptions.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run commands from the project root so `src` is importable.

## Dataset and metadata

Put uncovered raw images in `dataset/raw/uncovered/<roll_number>/` and covered raw images in `dataset/raw/covered/<roll_number>/`. The system discovers available images dynamically; equal counts and exactly 40 populated folders are not assumed. `dataset/metadata/students.csv` is authoritative for roll number, display name, face status, fingerprint ID, dataset folder, and active state.

The dataset contains images for all 40 requested rolls: 36 uncovered folders and 4 covered folders. Rolls were assigned sequentially using the original folder-entry order in `Data1.zip`; the authoritative mapping is `dataset/metadata/student_folder_mapping.csv`. Images use `<roll_number>_face_<three-digit serial>.<original_extension>` names.

## Experiment sequence

```bash
python scripts/01_preprocess_dataset.py
python scripts/02_extract_embeddings.py
python scripts/03_train_model.py
python scripts/04_evaluate_model.py
python scripts/05_test_single_image.py dataset/test_images/single/example.jpg
python scripts/06_test_group_image.py dataset/test_images/group/example.jpg
python scripts/07_test_video.py dataset/test_videos/classroom.mp4 --mode experiment
python scripts/08_calculate_attendance.py
python scripts/09_run_covered_face_tracking.py dataset/test_videos/classroom.mp4
python scripts/10_run_full_system.py FP001
```

`experiment` mode samples recorded video every 3 seconds. `realtime` mode samples every 180 seconds. Both values and the attendance threshold live only in `config/settings.yaml`.

## Research workflow

Preprocessing detects a face before cropping/resizing and copies failures to `dataset/rejected/`. Embeddings and labels are saved to `models/embeddings/embeddings.pkl`. Training uses a stratified split and writes the SVM, label encoder, held-out split, and model provenance. Evaluation writes metrics and classification artifacts into a named experiment without retraining.

The current preprocessing pass uses full-resolution RGB input with MTCNN, a 0.90 confidence threshold, 20% crop margin, aspect-preserving 160×160 resizing, conservative quality checks, and a recorded 90° fallback for sideways captures. Metadata-marked covered students are skipped and preserved for tracking. Activate `.venv` and rerun with `python scripts/01_preprocess_dataset.py`. Review `outputs/reports/preprocessing_report.md` before extracting embeddings.

Do not overwrite completed experiment directories. Create `exp_002_*`, record the dataset version and parameters, then generate plots from the saved CSV/JSON artifacts. Publication figures belong in `outputs/figures/paper/` and should be exported on a white background at 300 DPI; no experimental value should be typed manually into a plot.

## Current limitations

No prior application code or trained model existed to migrate. Hardware-specific fingerprint matching and production-grade multi-camera tracking require adapters for the selected scanner/camera. The included HOG/centroid tracker is an explicit baseline, not a claim of research-grade covered-person re-identification. The expensive 40-student preprocessing/training pipeline was intentionally not run during restructuring.
# SmartPresence
