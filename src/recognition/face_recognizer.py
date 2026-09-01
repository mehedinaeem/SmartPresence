"""Reusable MTCNN + FaceNet + trained-SVM local inference."""
from __future__ import annotations
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
import cv2, joblib, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from src.utils.paths import path, settings

@dataclass(frozen=True)
class Recognition:
    roll_number: str
    confidence: float
    x: int
    y: int
    w: int
    h: int
    detector_confidence: float
    def to_dict(self) -> dict: return asdict(self)

class FaceRecognizer:
    """Load artifacts once and recognize every MTCNN face in a BGR frame."""
    def __init__(self, classifier_path: Path|None=None, encoder_path: Path|None=None, unknown_threshold: float|None=None, initialize_facenet: bool=True) -> None:
        cfg=settings()["TESTING"]
        self.classifier_path=classifier_path or path("classifier"); self.encoder_path=encoder_path or path("label_encoder")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always"); self.classifier=joblib.load(self.classifier_path); self.encoder=joblib.load(self.encoder_path)
        self.load_warnings=[str(item.message) for item in caught]
        self.model_name=settings()["RECOGNITION"]["model_name"]; self.detector_backend=cfg["detector_backend"]; self.align=bool(cfg["align"]); self.normalization=cfg["normalization"]
        self.unknown_threshold=float(cfg["unknown_confidence_threshold"] if unknown_threshold is None else unknown_threshold)
        self.expected_dimension=int(settings()["RECOGNITION"]["embedding_dimension"])
        steps=dict(self.classifier.steps) if isinstance(self.classifier,Pipeline) else {}
        self.classifier_normalizes=any(isinstance(step,Normalizer) for step in steps.values())
        self._deepface=None
        if initialize_facenet: self.initialize_facenet()

    def initialize_facenet(self) -> None:
        if self._deepface is None:
            from deepface import DeepFace
            DeepFace.build_model(self.model_name); self._deepface=DeepFace

    def _predict(self, embedding: np.ndarray) -> tuple[str,float]:
        vector=np.asarray(embedding,dtype=np.float32).reshape(1,-1)
        if vector.shape[1]!=self.expected_dimension: raise ValueError(f"Expected {self.expected_dimension}-D embedding, received {vector.shape[1]}")
        if not self.classifier_normalizes: vector=vector/(np.linalg.norm(vector,axis=1,keepdims=True)+1e-12)
        probabilities=self.classifier.predict_proba(vector)[0]; index=int(np.argmax(probabilities)); confidence=float(probabilities[index])
        encoded_class=self.classifier.classes_[index] if hasattr(self.classifier,"classes_") else self.classifier.predict(vector)[0]
        roll=str(self.encoder.inverse_transform([int(encoded_class)])[0])
        return (roll if confidence>=self.unknown_threshold else "Unknown"),confidence

    def recognize_faces(self, frame_bgr: np.ndarray) -> list[Recognition]:
        if frame_bgr is None or not isinstance(frame_bgr,np.ndarray) or frame_bgr.size==0: raise ValueError("Frame is empty or unreadable")
        self.initialize_facenet()
        try:
            representations=self._deepface.represent(img_path=frame_bgr,model_name=self.model_name,detector_backend=self.detector_backend,enforce_detection=True,align=self.align,normalization=self.normalization)
        except ValueError as error:
            if "Face could not be detected" in str(error) or "face could not be detected" in str(error).lower(): return []
            raise
        results=[]
        for item in representations:
            area=item.get("facial_area",{}); x=max(0,int(area.get("x",0))); y=max(0,int(area.get("y",0))); w=max(0,int(area.get("w",0))); h=max(0,int(area.get("h",0)))
            roll,confidence=self._predict(np.asarray(item["embedding"],dtype=np.float32))
            results.append(Recognition(roll,confidence,x,y,w,h,float(item.get("face_confidence",0.0) or 0.0)))
        return results

    def model_summary(self) -> dict:
        steps=[(name,type(step).__name__) for name,step in self.classifier.steps] if isinstance(self.classifier,Pipeline) else []
        labels=[str(item) for item in self.encoder.classes_]
        return {"classifier_type":f"{type(self.classifier).__module__}.{type(self.classifier).__name__}","is_pipeline":isinstance(self.classifier,Pipeline),"pipeline_steps":steps,"classifier_normalizes":self.classifier_normalizes,"embedding_dimension":getattr(self.classifier,"n_features_in_",None),"class_count":len(labels),"class_labels":labels,"load_warnings":self.load_warnings}

def draw_results(frame: np.ndarray, detections: list[Recognition]) -> np.ndarray:
    annotated=frame.copy()
    for detection in detections:
        known=detection.roll_number!="Unknown"; color=(30,170,30) if known else (40,40,220)
        x,y,w,h=detection.x,detection.y,detection.w,detection.h; cv2.rectangle(annotated,(x,y),(x+w,y+h),color,2)
        label=f"{detection.roll_number} | {detection.confidence*100:.1f}%"; (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.55,2)
        top=max(0,y-th-10); cv2.rectangle(annotated,(x,top),(x+tw+8,y),color,-1); cv2.putText(annotated,label,(x+4,max(th+1,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
    return annotated
