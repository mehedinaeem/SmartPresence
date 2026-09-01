import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, cv2
from src.tracking.covered_face_tracker import CoveredFaceTracker

parser=argparse.ArgumentParser(); parser.add_argument("video")
if __name__ == "__main__":
    cap=cv2.VideoCapture(parser.parse_args().video); tracker=CoveredFaceTracker()
    while True:
        ok,frame=cap.read()
        if not ok: break
        print(tracker.process(frame))
    cap.release()
