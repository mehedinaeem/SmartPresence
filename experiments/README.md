# Research run provenance

Historical artifacts are preserved as recorded. `exp_001_baseline/metrics.json` is an empty migration placeholder, not a measured baseline. The imported classifier metadata lives in `models/classifiers/model_info.json`: 6,901 embeddings, 36 classes, 5,520 training and 1,381 test samples. The local preprocessing report/manifests describe 5,737 accepted crops. Current processed files must be inventoried separately; equal counts would not establish identical data or splits.

Each future extraction, training, or evaluation creates a unique directory and refuses an existing run ID:

```text
experiments/<run_id>/
  experiment_config.json     stage, timestamp, Git commit/dirty state, settings, environment
  dataset_manifest.json      sorted relative image paths, content hashes, cohort counts
  dataset_hash.txt            hash of canonical manifest file records
  README.md
  embeddings.pkl             extraction stage only
  face_classifier.pkl        training stage only
  label_encoder.pkl          training stage only
  evaluation_split.pkl       training stage only
  split_indices.json         training stage only: exact train/test indices and source ordering
  model_info.json             training stage only
  metrics.json               evaluation stage only
  classification_report.csv  evaluation stage only
  confusion_matrix.csv       evaluation stage only
  source_run.json             evaluation stage only: source artifact hashes
```

This is a template, not a claim that old runs contain these files. Incomplete/failed directories are retained for diagnosis; use a new ID for retries. No missing historical metrics are synthesized.

Example, explicitly initiated by the researcher:

```bash
python scripts/02_extract_embeddings.py --run-id embeddings_example
python scripts/03_train_model.py --embeddings experiments/embeddings_example/embeddings.pkl --run-id training_example
python scripts/04_evaluate_model.py --training-run experiments/training_example --run-id evaluation_example
```

Extraction fingerprints the actual processed image bytes and rechecks them after extraction. Training requires the embedded provenance and records the embedding artifact SHA-256, exact split, and current training configuration. Evaluation inherits the training configuration/dataset manifest and records its own execution environment and source hashes. Never load untrusted pickle/joblib artifacts.

The v1 dataset hash is SHA-256 of UTF-8 canonical JSON (`sort_keys=True`, separators `,` and `:`) containing sorted relative-path/content-SHA-256 records. This documented algorithm is not assumed to match the historical Colab hash algorithm. Preserve source metadata, acquisition groups, and consent records privately. An image-level stratified split does not establish independence between subjects, sessions, or near-duplicate frames; disclose the split unit in the paper.

Training matches the recorded methodological SVM configuration; exact historical numeric reproduction also requires the original dataset version, extraction settings, package versions, and split. The historical metadata does not explicitly state FaceNet input normalization. Future extraction records `Facenet` normalization, consistent with the existing local inference configuration; this does not retrospectively prove the Colab settings.
