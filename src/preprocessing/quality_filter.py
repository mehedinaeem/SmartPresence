"""Conservative face-crop quality measurements."""
from __future__ import annotations
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass(frozen=True)
class QualityResult:
    blur_score: float; brightness_score: float; rejection_reason: str | None

def evaluate_quality(face_bgr: np.ndarray, *, blur_check: bool, blur_threshold: float, brightness_check: bool, min_brightness: float, max_brightness: float) -> QualityResult:
    if not isinstance(face_bgr,np.ndarray) or face_bgr.size==0 or face_bgr.ndim!=3 or face_bgr.shape[2]!=3:
        return QualityResult(0.0,0.0,"invalid_output")
    gray=cv2.cvtColor(face_bgr,cv2.COLOR_BGR2GRAY)
    blur=float(cv2.Laplacian(gray,cv2.CV_64F).var()); brightness=float(gray.mean())
    if blur_check and blur < blur_threshold: return QualityResult(blur,brightness,"severe_blur")
    if brightness_check and brightness < min_brightness: return QualityResult(blur,brightness,"extremely_dark")
    if brightness_check and brightness > max_brightness: return QualityResult(blur,brightness,"overexposed")
    return QualityResult(blur,brightness,None)

def quality_reason(face: np.ndarray,min_sharpness: float=20.0)->str|None:
    return evaluate_quality(face,blur_check=True,blur_threshold=min_sharpness,brightness_check=True,min_brightness=20,max_brightness=245).rejection_reason
