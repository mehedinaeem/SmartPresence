import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.recognition.face_recognizer import FaceRecognizer

if __name__=="__main__":
    recognizer=FaceRecognizer(initialize_facenet=False); summary=recognizer.model_summary(); expected=[str(n) for n in range(22102001,22102041)]; actual=summary["class_labels"]
    print(json.dumps(summary,indent=2)); missing=sorted(set(expected)-set(actual)); unexpected=sorted(set(actual)-set(expected))
    print(f"\nExpected classes: 40\nDetected classes: {len(actual)}\nExpected range: {expected[0]} - {expected[-1]}")
    if actual==expected: print("MODEL LABEL CHECK: PASS")
    else:
        print("MODEL LABEL CHECK: WARNING")
        if missing: print("Missing labels:",", ".join(missing))
        if unexpected: print("Unexpected labels:",", ".join(unexpected))
    if recognizer.load_warnings: print("Compatibility warnings:\n- " + "\n- ".join(recognizer.load_warnings))
