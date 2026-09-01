# SmartPresence migration report

> Dataset replacement notice (latest): the earlier in-project assignment was superseded by the verified ZIP-entry order from `/home/mehedinaeem/Downloads/Data1.zip`. The authoritative mapping is now `dataset/metadata/student_folder_mapping.csv`. All image files were renamed to `<roll>_face_<serial>.<original_extension>`, and the previous raw/processed image dataset was removed after verification.

## Source inventory

The original repository contained only `Data/` with 7,464 JPEG images (about 324 MB). It contained no source code, notebooks, models, embeddings, videos, logs, configuration, dependency file, README, or Git history.

## Migration mapping

- `Data/2210201` → `dataset/raw/uncovered/22102001`
- `Data/2210202` → `dataset/raw/uncovered/22102002`
- `Data/2210204..flip` → `dataset/raw/uncovered/22102004`
- `Data/2210206` → `dataset/raw/uncovered/22102006`
- `Data/2210208` → `dataset/raw/uncovered/22102008`
- `Data/2210209` → `dataset/raw/uncovered/22102009`
- `Data/22102036_flip` → `dataset/raw/uncovered/22102036`
- Other valid folders from 22102001–22102040 → matching folder under `dataset/raw/uncovered/`
- `Data/Covered`, `Covered1`, `covered3`, `covered4` → `dataset/archive/legacy_unmapped/` because no roll-number mapping was available
- Folders with roll numbers above 22102040 → `dataset/archive/out_of_cohort/`

No images were deleted. The active cohort area contains 4,414 images for 25 roll folders; the archives contain 3,050 preserved images.

## Exact duplicates retained

- `22102018/frame_0070.jpg` = `22102018/frame_0100.jpg`
- `22102018/frame_0084.jpg` = `22102018/frame_0120.jpg`
- `22102018/frame_0077.jpg` = `22102018/frame_0110.jpg`

Recommended action: verify acquisition provenance and then retain one file from each pair. Nothing was removed during migration.

## Items requiring human confirmation

- Confirm that shortened labels `2210201`, `2210202`, `2210206`, `2210208`, and `2210209` mean rolls 22102001, 02, 06, 08, and 09.
- Confirm that `2210204..flip` and `22102036_flip` belong to rolls 22102004 and 22102036.
- Associate the four anonymous covered folders with roll numbers before moving them into `dataset/raw/covered/`.
- Decide whether out-of-cohort rolls are relevant to another experiment.
- Replace placeholder student names, face visibility values, and fingerprint IDs with verified study metadata.

## Final 40-roll reassignment

At the user's direction, every previously out-of-range or anonymous folder was used to fill a missing roll from 22102001–22102040. Numeric source folders and missing rolls were paired in ascending order; covered sources were paired with the remaining missing rolls in their displayed folder order.

- `22102042` → `22102005`
- `22102043` → `22102007`
- `22102044` → `22102010`
- `22102046` → `22102011`
- `22102048` → `22102015`
- `22102050` → `22102017`
- `22102055` → `22102019`
- `22102061` → `22102020`
- `22102067` → `22102026`
- `22102070` → `22102028`
- `22102071` → `22102031`
- `Covered` → `22102032`
- `Covered1` → `22102033`
- `covered3` → `22102035`
- `covered4` → `22102038`

The active raw dataset now contains exactly 40 roll folders: 36 uncovered and 4 covered. All 7,464 images remain preserved.
