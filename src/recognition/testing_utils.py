"""Shared local-testing I/O and reporting helpers."""
from __future__ import annotations
import csv,json
from datetime import datetime,timezone
from pathlib import Path
from src.utils.paths import PROJECT_ROOT,path,settings

IMAGE_EXTENSIONS={".jpg",".jpeg",".png",".bmp",".webp"}; VIDEO_EXTENSIONS={".mp4",".avi",".mov",".mkv"}

def supported_files(folder: Path,extensions: set[str])->list[Path]:
    folder.mkdir(parents=True,exist_ok=True); return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions)

def write_csv(destination: Path,fieldnames:list[str],rows:list[dict])->None:
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fieldnames); writer.writeheader(); writer.writerows({key:row.get(key,"") for key in fieldnames} for row in rows)

def save_test_config(class_count:int)->Path:
    cfg=settings()["TESTING"]
    payload={"facenet_model":settings()["RECOGNITION"]["model_name"],"detector":cfg["detector_backend"],"align":cfg["align"],"normalization":cfg["normalization"],"unknown_confidence_threshold":cfg["unknown_confidence_threshold"],"video_frame_interval_seconds":cfg["video_frame_interval_seconds"],"attendance_threshold":cfg["attendance_threshold"],"number_of_classes":class_count,"model_file":str(path("classifier").relative_to(PROJECT_ROOT)),"label_encoder_file":str(path("label_encoder").relative_to(PROJECT_ROOT)),"test_date":datetime.now(timezone.utc).isoformat()}
    destination=PROJECT_ROOT/"outputs/reports/test_config.json"; destination.parent.mkdir(parents=True,exist_ok=True); destination.write_text(json.dumps(payload,indent=2),encoding="utf-8"); return destination
