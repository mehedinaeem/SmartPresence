# Model artifacts

The existing imported Colab artifacts are not modified by future training commands:

- `classifiers/face_classifier.pkl`: 36-class FaceNet-SVM recognition model within a 40-participant SmartPresence evaluation; 128-D input, L2 Normalizer + RBF SVC, C=10, gamma=scale, balanced class weights.
- `classifiers/label_encoder.pkl`: maps the 36 uncovered classes to enrolled roll numbers.
- `classifiers/model_info.json`: original run provenance, counts and recorded metrics. The 6,901-embedding snapshot is distinct from the 5,737-crop local preprocessing report.
- `tracking/yolo11n.pt`: pretrained person detection for consecutive-frame ByteTrack tracking. Track IDs are temporary, not student identities; track 7 does not imply roll 22102007.
- `metadata/model_info.json`: retained historical empty placeholder, not authoritative classifier metadata.

New training artifacts go to `experiments/<run_id>/`. They do not become the default inference model automatically. To evaluate a future model, select its run explicitly with `04_evaluate_model.py`; promotion to default inference paths is a separate, intentional researcher action.

Use scikit-learn 1.6.1 for serialized classifier compatibility. Load only trusted artifacts. Model binaries may be unavailable in a fresh code-only checkout; obtain authorized originals and record hashes instead of silently substituting models. The repository validator checks metadata and file existence without executing pickle payloads or downloading weights.
