import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse, cv2
from src.preprocessing.face_detector import FaceDetector
from src.preprocessing.face_cropper import crop_face
from src.recognition.face_recognizer import FaceRecognizer
from src.utils.paths import settings

parser=argparse.ArgumentParser(); parser.add_argument("image")
if __name__ == "__main__":
    args=parser.parse_args(); image=cv2.imread(args.image)
    if image is None: raise FileNotFoundError(args.image)
    boxes=FaceDetector().detect(image)
    if not boxes: raise RuntimeError("No face detected")
    cfg=settings(); face=crop_face(image,max(boxes,key=lambda b:b[2]*b[3]),cfg["FACE_DETECTION"]["margin"],(cfg["FACE_SIZE"]["width"],cfg["FACE_SIZE"]["height"]))
    print(dict(zip(("roll_number","confidence"),FaceRecognizer().recognize(face))))
