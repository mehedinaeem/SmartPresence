# SmartPresence preprocessing report

- Run: MTCNN preprocessing only; no FaceNet embeddings or SVM training
- Raw images: 7,464
- Covered images skipped: 800 across rolls 22102002, 22102030, 22102031, and 22102033
- Eligible uncovered images: 6,664
- Processed 160×160 faces: 5,737
- Rejected images: 927
- Eligible-image success rate: 86.09%
- Mean processed images per uncovered student: 159.36
- Median: 160
- Minimum: 47 (22102016)
- Maximum: 341 (22102038)

## Rejection reasons

- No face detected: 640
- Detection confidence below 0.90: 255
- Face too small: 31
- Multiple ambiguous faces: 1
- Corrupted images: 0
- Severe blur/illumination failures: 0

The single ambiguous multiple-face sample is `22102029_face_165.jpg`; it was rejected to prevent label contamination.

## Orientation recovery

Sixty-eight sideways source images were recovered using the configured 90° clockwise MTCNN fallback. Roll 22102019 improved from 2/57 to 57/57 without lowering the 0.90 threshold. Rotation is recorded per image in `processed_images.csv`; raw files remain unchanged.

## Balance warnings

Classes below half the uncovered-class mean are 22102005 (58), 22102009 (73), 22102011 (73), 22102014 (53), 22102016 (47), 22102019 (57), and 22102020 (54). These low totals mainly reflect the original collection counts, not preprocessing failure. No images were duplicated or augmented.

## Validation

All processed files open successfully, are exactly 160×160 with three color channels, reside under the matching roll folder, and have unique output paths. Processed and rejected metadata row counts match the filesystem. The raw SHA-256 fingerprint after processing matches the pre-run fingerprint recorded in `preprocessing_config.json`.
