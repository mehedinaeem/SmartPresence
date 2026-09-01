import sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.recognition.face_recognizer import FaceRecognizer,draw_results
from src.recognition.testing_utils import VIDEO_EXTENSIONS,save_test_config,supported_files,write_csv
from src.utils.paths import PROJECT_ROOT,path,settings

EXPECTED_ROLLS=[str(n) for n in range(22102001,22102041)]

def presence_chart(rows:list[dict],destination:Path,threshold_percent:float)->None:
    fig,ax=plt.subplots(figsize=(14,6)); ax.bar([r["roll_number"][-2:] for r in rows],[r["presence_percentage"] for r in rows],color="#32689b")
    ax.axhline(threshold_percent,color="#b22222",linestyle="--",linewidth=1.5,label=f"{threshold_percent:.0f}% threshold"); ax.set_xlabel("Student roll suffix"); ax.set_ylabel("Presence (%)"); ax.set_ylim(0,105); ax.legend(); fig.tight_layout(); destination.parent.mkdir(parents=True,exist_ok=True); fig.savefig(destination,dpi=300,facecolor="white"); plt.close(fig)

if __name__=="__main__":
    cfg=settings()["TESTING"]; interval=float(cfg["video_frame_interval_seconds"]); threshold=float(cfg["attendance_threshold"]); threshold_percent=threshold*100
    source=path("test_videos"); video_output=PROJECT_ROOT/"outputs/videos"; attendance_output=PROJECT_ROOT/"outputs/attendance"; video_output.mkdir(parents=True,exist_ok=True); attendance_output.mkdir(parents=True,exist_ok=True)
    recognizer=FaceRecognizer(); frame_rows=[]; attendance_rows=[]; videos=supported_files(source,VIDEO_EXTENSIONS); print(f"Video files: {len(videos)}")
    for video in videos:
        capture=cv2.VideoCapture(str(video)); fps=float(capture.get(cv2.CAP_PROP_FPS)); total_frames=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or fps<=0 or total_frames<=0: print(f"WARNING invalid video: {video}"); capture.release(); continue
        step=max(1,round(fps*interval)); sample_frames=list(range(0,total_frames,step)); output_file=video_output/f"{video.stem}_marked.mp4"; width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer=cv2.VideoWriter(str(output_file),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height)); counts={roll:0 for roll in EXPECTED_ROLLS}
        if not writer.isOpened(): capture.release(); print(f"ERROR cannot create video: {output_file}"); continue
        for sample_number,frame_number in enumerate(sample_frames,1):
            capture.set(cv2.CAP_PROP_POS_FRAMES,frame_number); ok,frame=capture.read(); timestamp=frame_number/fps
            if not ok or frame is None:
                frame_rows.append({"video_name":video.name,"sample_number":sample_number,"source_frame_number":frame_number,"timestamp_seconds":timestamp,"roll_number":"","confidence":"","status":"corrupt_frame"}); continue
            try: detections=recognizer.recognize_faces(frame)
            except Exception as error:
                print(f"ERROR {video.name} sample {sample_number}: {error}"); detections=[]; frame_rows.append({"video_name":video.name,"sample_number":sample_number,"source_frame_number":frame_number,"timestamp_seconds":timestamp,"roll_number":"","confidence":"","status":"recognition_error"})
            best={}
            for detection in detections:
                if detection.roll_number!="Unknown" and (detection.roll_number not in best or detection.confidence>best[detection.roll_number].confidence): best[detection.roll_number]=detection
                frame_rows.append({"video_name":video.name,"sample_number":sample_number,"source_frame_number":frame_number,"timestamp_seconds":f"{timestamp:.3f}","roll_number":detection.roll_number,"confidence":f"{detection.confidence:.6f}","status":"recognized" if detection.roll_number!="Unknown" else "unknown"})
            if not detections: frame_rows.append({"video_name":video.name,"sample_number":sample_number,"source_frame_number":frame_number,"timestamp_seconds":f"{timestamp:.3f}","roll_number":"","confidence":"","status":"no_face"})
            for roll in best:
                if roll in counts: counts[roll]+=1
            annotated=draw_results(frame,detections); overlay=f"Sample #{sample_number} | Time: {timestamp:.1f}s"; cv2.putText(annotated,overlay,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),4,cv2.LINE_AA); cv2.putText(annotated,overlay,(15,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
            repeats=min(step,total_frames-frame_number)
            for _ in range(repeats): writer.write(annotated)
            print(f"[{video.name}] sample {sample_number}/{len(sample_frames)} at {timestamp:.1f}s: {sorted(best)}")
        capture.release(); writer.release(); total_samples=len(sample_frames); current=[]
        for roll in EXPECTED_ROLLS:
            percentage=100*counts[roll]/total_samples if total_samples else 0; row={"video_name":video.name,"roll_number":roll,"detected_sampled_frames":counts[roll],"total_sampled_frames":total_samples,"presence_percentage":round(percentage,2),"attendance_threshold":threshold_percent,"status":"Present" if percentage>=threshold_percent else "Absent"}; attendance_rows.append(row); current.append(row)
        chart_name="presence_percentage.png" if len(videos)==1 else f"{video.stem}_presence_percentage.png"; presence_chart(current,attendance_output/chart_name,threshold_percent)
        print(f"\nSmartPresence Video Test Complete\nVideo: {video.name}\nSampling: Every {interval:g} seconds\nTotal sampled frames: {total_samples}\nAttendance threshold: {threshold_percent:g}%")
        print("Roll       Detected  Presence  Status")
        for row in current: print(f"{row['roll_number']}   {row['detected_sampled_frames']:>4}      {row['presence_percentage']:>6.2f}%  {row['status']}")
    write_csv(attendance_output/"frame_presence_log.csv",["video_name","sample_number","source_frame_number","timestamp_seconds","roll_number","confidence","status"],frame_rows)
    write_csv(attendance_output/"final_attendance.csv",["video_name","roll_number","detected_sampled_frames","total_sampled_frames","presence_percentage","attendance_threshold","status"],attendance_rows); save_test_config(len(recognizer.encoder.classes_))
