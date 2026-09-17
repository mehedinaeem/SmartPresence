"""Show configured research status without loading vision models."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.paths import path, settings
from src.utils.research import students

if __name__ == "__main__":
    cfg, cohort = settings(), students()
    print("SmartPresence Research Status\n")
    for label, value in {
        "Participants": len(cohort),
        "Face-recognition participants": sum(s["face_status"] == "uncovered" for s in cohort),
        "Covered tracking participants": sum(s["face_status"] == "covered" for s in cohort),
        "Saved classifier": "Available" if path("classifier").is_file() else "Missing",
        "FaceNet dimension": cfg["RECOGNITION"]["embedding_dimension"],
        "Future classifier": f"{cfg['RECOGNITION']['normalization']} Normalizer + {cfg['RECOGNITION']['svm_kernel']} SVM",
        "Attendance threshold": f"{cfg['ATTENDANCE_THRESHOLD']:.0%}",
        "Recorded-video sampling (s)": cfg["EXPERIMENT_FRAME_INTERVAL_SECONDS"],
        "Live CCTV sampling (s)": cfg["REALTIME_CCTV_FRAME_INTERVAL_SECONDS"],
        "Body tracker": f"{Path(cfg['TRACKING']['model']).name} + {cfg['TRACKING']['tracker']}; every frame",
        "Fingerprint": "Static simulated identity mapping",
    }.items():
        print(f"{label}: {value}")
