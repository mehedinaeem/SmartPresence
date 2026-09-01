import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from src.recognition.face_recognizer import FaceRecognizer,draw_results
from src.recognition.testing_utils import IMAGE_EXTENSIONS,save_test_config,supported_files,write_csv
from src.utils.paths import PROJECT_ROOT,path

if __name__=="__main__":
    source=path("test_group_images"); destination=PROJECT_ROOT/"outputs/images/group"; destination.mkdir(parents=True,exist_ok=True); rows=[]; recognizer=FaceRecognizer()
    files=supported_files(source,IMAGE_EXTENSIONS); print(f"Group-image files: {len(files)}")
    for file in files:
        image=cv2.imread(str(file))
        if image is None: print(f"WARNING unreadable: {file}"); continue
        try: detections=recognizer.recognize_faces(image)
        except Exception as error: print(f"ERROR {file.name}: {error}"); continue
        print(f"\n{file.name}\nDetected:")
        for index,detection in enumerate(detections,1):
            item=detection.to_dict(); rows.append({"source_image":file.name,"face_index":index,**item}); print(f"{detection.roll_number} - {detection.confidence:.4f}")
        if not detections: print("No faces")
        cv2.imwrite(str(destination/f"{file.stem}_annotated{file.suffix}"),draw_results(image,detections))
    write_csv(PROJECT_ROOT/"outputs/reports/group_image_results.csv",["source_image","face_index","roll_number","confidence","detector_confidence","x","y","w","h"],rows); save_test_config(len(recognizer.encoder.classes_))
