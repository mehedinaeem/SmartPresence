"""Reproducible MTCNN preprocessing for the SmartPresence face dataset."""
from __future__ import annotations
import csv, hashlib, json, random, shutil, statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm
from src.preprocessing.face_cropper import crop_and_resize_face
from src.preprocessing.face_detector import MTCNNFaceDetector, select_primary_face
from src.preprocessing.quality_filter import evaluate_quality
from src.utils.helpers import image_files
from src.utils.paths import PROJECT_ROOT, path, settings

PROCESSED_FIELDS=["roll_number","original_filename","processed_filename","original_path","processed_path","original_width","original_height","rotation_degrees","face_x","face_y","face_width","face_height","detection_confidence","face_count","blur_score","brightness_score","status"]
REJECTED_FIELDS=["roll_number","original_filename","original_path","rejected_path","rejection_reason","face_count","detection_confidence","face_width","face_height","blur_score","brightness_score"]

def _raw_fingerprint(roots: list[Path]) -> str:
    digest=hashlib.sha256()
    for file in sorted(p for root in roots for p in root.rglob("*") if p.is_file()):
        digest.update(str(file.relative_to(PROJECT_ROOT)).encode()); digest.update(file.read_bytes())
    return digest.hexdigest()

def _load_students() -> list[dict[str,str]]:
    with path("students_metadata").open(newline="",encoding="utf-8") as stream: rows=list(csv.DictReader(stream))
    expected=[str(n) for n in range(22102001,22102041)]
    if [r["roll_number"] for r in rows if r.get("active","1")=="1"] != expected: raise RuntimeError("Active metadata must contain rolls 22102001 through 22102040 in order")
    return rows

def _write_csv(destination: Path, fields: list[str], rows: list[dict]) -> None:
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)

def _diagnostic_figures(summary: list[dict], reasons: Counter, samples: list[Path]) -> None:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output=PROJECT_ROOT/"outputs/figures/dataset"; output.mkdir(parents=True,exist_ok=True)
    rows=[r for r in summary if r["roll_number"]!="TOTAL"]
    for key,name in (("raw_images","dataset_distribution_raw.png"),("processed_images","dataset_distribution_processed.png")):
        fig,ax=plt.subplots(figsize=(14,5)); ax.bar([r["roll_number"][-2:] for r in rows],[int(r[key]) for r in rows],color="#32689b")
        ax.set_xlabel("Student roll suffix"); ax.set_ylabel("Images"); fig.tight_layout(); fig.savefig(output/name,dpi=300,facecolor="white"); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); labels=list(reasons) or ["none"]; values=[reasons[x] for x in labels] if reasons else [0]
    ax.bar(labels,values,color="#a64b43"); ax.tick_params(axis="x",rotation=35); ax.set_ylabel("Rejected images"); fig.tight_layout(); fig.savefig(output/"rejection_reasons.png",dpi=300,facecolor="white"); plt.close(fig)
    chosen=random.Random(42).sample(samples,min(25,len(samples)))
    if chosen:
        fig,axes=plt.subplots(5,5,figsize=(10,10));
        for ax in axes.flat: ax.axis("off")
        for ax,file in zip(axes.flat,chosen):
            image=cv2.cvtColor(cv2.imread(str(file)),cv2.COLOR_BGR2RGB); ax.imshow(image); ax.set_title(file.parent.name,fontsize=8); ax.axis("off")
        fig.tight_layout(); fig.savefig(output/"processed_samples_grid.png",dpi=200,facecolor="white"); plt.close(fig)

def preprocess_dataset() -> dict:
    cfg=settings()["PREPROCESSING"]; students=_load_students()
    roots={"uncovered":path("dataset_raw_uncovered"),"covered":path("dataset_raw_covered")}
    actual={p.name for root in roots.values() for p in root.iterdir() if p.is_dir()}; expected={str(n) for n in range(22102001,22102041)}
    if actual!=expected: raise RuntimeError(f"Raw folders mismatch. Missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}")
    before=_raw_fingerprint(list(roots.values())); stage=PROJECT_ROOT/"dataset/.preprocessing_stage"
    if stage.exists(): shutil.rmtree(stage)
    processed_stage,rejected_stage=stage/"processed",stage/"rejected"; processed_stage.mkdir(parents=True); rejected_stage.mkdir(parents=True)
    detector=MTCNNFaceDetector(); processed_rows=[]; rejected_rows=[]; summary=[]; reasons=Counter(); valid_counts=[]; samples=[]
    size=(int(cfg["face_size"]["width"]),int(cfg["face_size"]["height"])); padding=tuple(cfg["padding_value"]); batch_size=int(cfg.get("batch_size",16))
    for student in students:
        roll=student["roll_number"]; status=student["face_status"].strip().lower(); source=roots[status]/roll
        files=image_files(source); (processed_stage/roll).mkdir(); (rejected_stage/roll).mkdir()
        if status=="covered":
            summary.append({"roll_number":roll,"raw_images":len(files),"processed_images":0,"rejected_images":0,"skipped_covered_images":len(files),"processing_success_rate":"0.00"}); valid_counts.append(0); print(f"[{roll}] Skipped {len(files)} covered-face images")
            continue
        valid=0; rejected=0; bar=tqdm(total=len(files),desc=f"[{roll}]",unit="image")
        for start in range(0,len(files),batch_size):
            batch=files[start:start+batch_size]; loaded=[]; loaded_paths=[]
            for source_file in batch:
                image=cv2.imread(str(source_file),cv2.IMREAD_COLOR)
                if image is None or image.size==0 or image.ndim!=3 or image.shape[2]!=3:
                    target=rejected_stage/roll/source_file.name; shutil.copy2(source_file,target); rejected+=1; reasons["corrupted_image"]+=1
                    rejected_rows.append({"roll_number":roll,"original_filename":source_file.name,"original_path":str(source_file.relative_to(PROJECT_ROOT)),"rejected_path":str(Path("dataset/rejected")/roll/target.name),"rejection_reason":"corrupted_image","face_count":0,"detection_confidence":"","face_width":"","face_height":"","blur_score":"","brightness_score":""}); bar.update(1); continue
                loaded.append((image,cv2.cvtColor(image,cv2.COLOR_BGR2RGB))); loaded_paths.append(source_file)
            try: detections_batch=detector.detect_batch([rgb for _,rgb in loaded])
            except Exception as error:
                detections_batch=[error]*len(loaded)
            for source_file,(image,_),detections in zip(loaded_paths,loaded,detections_batch):
                reason=None; selected=None; face_count=0; confidence=""; face_width=""; face_height=""; blur=""; brightness=""; rotation=0; working_image=image
                try:
                    if isinstance(detections,Exception): raise detections
                    selected,reason,face_count=select_primary_face(detections,float(cfg["detection_confidence"]),int(cfg["min_face_width"]),int(cfg["min_face_height"]),float(cfg.get("multiple_face_dominance_ratio",1.75)))
                    if not selected and reason in {"no_face_detected","detection_confidence_below_threshold","face_too_small"}:
                        for degrees in cfg.get("orientation_fallback_degrees",[]):
                            rotation_code={90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}[int(degrees)]
                            candidate=cv2.rotate(image,rotation_code)
                            fallback=detector.detect(cv2.cvtColor(candidate,cv2.COLOR_BGR2RGB))
                            fallback_selected,fallback_reason,fallback_count=select_primary_face(fallback,float(cfg["detection_confidence"]),int(cfg["min_face_width"]),int(cfg["min_face_height"]),float(cfg.get("multiple_face_dominance_ratio",1.75)))
                            if fallback_selected:
                                selected,reason,face_count=fallback_selected,None,fallback_count; rotation=int(degrees); working_image=candidate; break
                    if selected:
                        confidence=selected.confidence; face_width=selected.width; face_height=selected.height
                        face,crop_box,_=crop_and_resize_face(working_image,selected.box,float(cfg["face_margin"]),size,padding)
                        quality=evaluate_quality(face,blur_check=bool(cfg["blur_check"]),blur_threshold=float(cfg["blur_threshold"]),brightness_check=bool(cfg["brightness_check"]),min_brightness=float(cfg["min_brightness"]),max_brightness=float(cfg["max_brightness"]))
                        blur=quality.blur_score; brightness=quality.brightness_score; reason=quality.rejection_reason
                except Exception as error: reason=f"processing_error:{type(error).__name__}"
                if reason:
                    target=rejected_stage/roll/source_file.name; shutil.copy2(source_file,target); rejected+=1; reasons[reason]+=1
                    rejected_rows.append({"roll_number":roll,"original_filename":source_file.name,"original_path":str(source_file.relative_to(PROJECT_ROOT)),"rejected_path":str(Path("dataset/rejected")/roll/target.name),"rejection_reason":reason,"face_count":face_count,"detection_confidence":confidence,"face_width":face_width,"face_height":face_height,"blur_score":blur,"brightness_score":brightness})
                else:
                    valid+=1; filename=f"{roll}_face_{valid:03d}.jpg"; target=processed_stage/roll/filename
                    if target.exists(): raise RuntimeError(f"Refusing overwrite: {target}")
                    if not cv2.imwrite(str(target),face,[cv2.IMWRITE_JPEG_QUALITY,95]): raise RuntimeError(f"Could not save {target}")
                    x1,y1,x2,y2=crop_box; samples.append(target)
                    processed_rows.append({"roll_number":roll,"original_filename":source_file.name,"processed_filename":filename,"original_path":str(source_file.relative_to(PROJECT_ROOT)),"processed_path":str(Path("dataset/processed")/roll/filename),"original_width":image.shape[1],"original_height":image.shape[0],"rotation_degrees":rotation,"face_x":selected.x,"face_y":selected.y,"face_width":selected.width,"face_height":selected.height,"detection_confidence":selected.confidence,"face_count":face_count,"blur_score":blur,"brightness_score":brightness,"status":"processed"})
                bar.update(1)
        bar.close(); rate=100*valid/len(files) if files else 0; valid_counts.append(valid)
        summary.append({"roll_number":roll,"raw_images":len(files),"processed_images":valid,"rejected_images":rejected,"skipped_covered_images":0,"processing_success_rate":f"{rate:.2f}"}); print(f"[{roll}] Completed: {valid} valid, {rejected} rejected")
    totals={key:sum(int(r[key]) for r in summary) for key in ("raw_images","processed_images","rejected_images","skipped_covered_images")}
    eligible_raw=totals["raw_images"]-totals["skipped_covered_images"]; total_rate=100*totals["processed_images"]/eligible_raw if eligible_raw else 0
    summary.append({"roll_number":"TOTAL",**totals,"processing_success_rate":f"{total_rate:.2f}"})
    metadata=stage/"metadata"; _write_csv(metadata/"processed_images.csv",PROCESSED_FIELDS,processed_rows); _write_csv(metadata/"rejected_images.csv",REJECTED_FIELDS,rejected_rows)
    _write_csv(metadata/"preprocessing_summary.csv",["roll_number","raw_images","processed_images","rejected_images","skipped_covered_images","processing_success_rate"],summary)
    if _raw_fingerprint(list(roots.values()))!=before: raise RuntimeError("Raw dataset changed during preprocessing")
    for file in processed_stage.rglob("*.jpg"):
        check=cv2.imread(str(file));
        if check is None or check.shape[:2]!=(size[1],size[0]): raise RuntimeError(f"Invalid processed output: {file}")
    if len(processed_rows)!=sum(1 for p in processed_stage.rglob("*.jpg")): raise RuntimeError("Processed metadata count mismatch")
    if len(rejected_rows)!=sum(1 for p in rejected_stage.rglob("*") if p.is_file()): raise RuntimeError("Rejected metadata count mismatch")
    for name in ("processed","rejected"):
        destination=PROJECT_ROOT/"dataset"/name
        if destination.exists(): shutil.rmtree(destination)
        shutil.move(str(stage/name),destination)
    for name in ("processed_images.csv","rejected_images.csv","preprocessing_summary.csv"): shutil.move(str(metadata/name),PROJECT_ROOT/"dataset/metadata"/name)
    shutil.rmtree(stage)
    snapshot={"detector":"mtcnn","face_size":size,"face_margin":cfg["face_margin"],"detection_confidence":cfg["detection_confidence"],"minimum_face_size":[cfg["min_face_width"],cfg["min_face_height"]],"orientation_fallback_degrees":cfg.get("orientation_fallback_degrees",[]),"blur_threshold":cfg["blur_threshold"],"brightness_range":[cfg["min_brightness"],cfg["max_brightness"]],"processing_date":datetime.now(timezone.utc).isoformat(),"students_total":40,"students_processed":36,"students_skipped_covered":4,"raw_sha256":before}
    report=PROJECT_ROOT/"outputs/reports/preprocessing_config.json"; report.parent.mkdir(parents=True,exist_ok=True); report.write_text(json.dumps(snapshot,indent=2),encoding="utf-8")
    final_samples=[PROJECT_ROOT/str(p.relative_to(stage)).replace("processed/","dataset/processed/",1) for p in samples]
    _diagnostic_figures(summary,reasons,final_samples)
    uncovered_counts=[int(r["processed_images"]) for r in summary[:-1] if int(r["skipped_covered_images"])==0]
    result={"students_detected":40,"students_processed":36,"covered_students_skipped":4,"total_raw":totals["raw_images"],"eligible_raw":eligible_raw,"processed":totals["processed_images"],"rejected":totals["rejected_images"],"success_rate":total_rate,"mean":statistics.mean(uncovered_counts),"median":statistics.median(uncovered_counts),"minimum":min(uncovered_counts),"maximum":max(uncovered_counts),"rejection_reasons":dict(reasons)}
    print("\nSmartPresence Dataset Preprocessing Complete\n"+json.dumps(result,indent=2)); return result
