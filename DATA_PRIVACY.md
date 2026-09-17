# Data privacy and release review

Face images are biometric/personal data. Public release requires participant consent and institutional/ethical permission where applicable. This repository does not assert that ethics approval or permission to publish participant images has been obtained.

Publish code and reviewed derived metadata unless image sharing is explicitly authorized. Roll numbers, per-person predictions, attendance logs, crops, embeddings, annotated videos, and even filenames may identify people; review or pseudonymize them before release. Keep consent and identity linkage records in controlled storage.

`.gitignore` excludes future raw/processed/rejected images, private test data, ZIP archives and model/embedding pickle files. Ignore rules do not remove previously tracked files or erase Git history. The cleanup audit found 6,901 tracked processed images. No images were deleted or untracked automatically. The owner must review release authorization and, if needed, arrange removal from the Git index and history, existing releases, forks and shared archives before publication.

Review `git ls-files dataset/processed` and the validation warnings before pushing. Inspect output images/videos and participant-level CSVs separately; an aggregate research metric is not permission to publish underlying records. Code licensing and participant-data permissions are separate decisions.
