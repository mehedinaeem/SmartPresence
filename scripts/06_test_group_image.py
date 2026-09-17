"""Recognize faces through the existing MTCNN/FaceNet/SVM inference API."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    import cv2
    from src.recognition.face_recognizer import FaceRecognizer
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    detections = FaceRecognizer().recognize_faces(frame)
    if not detections:
        print("No faces detected")
    for result in detections:
        print(result.to_dict())
