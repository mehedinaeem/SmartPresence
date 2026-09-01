import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from src.recognition.face_recognizer import FaceRecognizer,draw_results
from src.recognition.testing_utils import IMAGE_EXTENSIONS,save_test_config,supported_files,write_csv
from src.utils.paths import PROJECT_ROOT,path

if __name__=="__main__":
    source=path("test_single_images"); destination=PROJECT_ROOT/"outputs/images/single"; destination.mkdir(parents=True,exist_ok=True); rows=[]; recognizer=FaceRecognizer()
    files=supported_files(source,IMAGE_EXTENSIONS); print(f"Single-image files: {len(files)}")
    for file in files:
        image=cv2.imread(str(file))
        if image is None: print(f"WARNING unreadable: {file}"); continue
        try: detections=recognizer.recognize_faces(image)
        except Exception as error: print(f"ERROR {file.name}: {error}"); continue
        for detection in detections:
            item=detection.to_dict(); rows.append({"source_image":file.name,**item}); print(f"{file.name}: {detection.roll_number} ({detection.confidence:.4f})")
        cv2.imwrite(str(destination/f"{file.stem}_annotated{file.suffix}"),draw_results(image,detections))
    write_csv(PROJECT_ROOT/"outputs/reports/single_image_results.csv",["source_image","roll_number","confidence","x","y","w","h"],rows); save_test_config(len(recognizer.encoder.classes_))
