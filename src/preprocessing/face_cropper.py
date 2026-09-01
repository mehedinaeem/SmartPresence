"""Validated, margin-aware face cropping."""
from __future__ import annotations
import numpy as np
from src.preprocessing.resize_face import ResizeMetadata, resize_face_crop

def expanded_box(image_shape: tuple[int, ...], box: tuple[int, int, int, int], margin_fraction: float) -> tuple[int, int, int, int]:
    x, y, width, height = box; image_height, image_width = image_shape[:2]
    if width <= 0 or height <= 0: raise ValueError("Invalid bounding box")
    margin_x, margin_y = round(width * margin_fraction), round(height * margin_fraction)
    x1, y1 = max(0, x-margin_x), max(0, y-margin_y)
    x2, y2 = min(image_width, x+width+margin_x), min(image_height, y+height+margin_y)
    if x2 <= x1 or y2 <= y1: raise ValueError("Expanded bounding box is outside image")
    return x1, y1, x2, y2

def crop_and_resize_face(image_bgr: np.ndarray, box: tuple[int, int, int, int], margin_fraction: float, size: tuple[int, int], padding_value: tuple[int, int, int]=(0,0,0)) -> tuple[np.ndarray, tuple[int,int,int,int], ResizeMetadata]:
    x1,y1,x2,y2 = expanded_box(image_bgr.shape, box, margin_fraction)
    crop=image_bgr[y1:y2,x1:x2]
    if crop.size == 0: raise ValueError("Face crop is empty")
    resized, metadata=resize_face_crop(crop,size,padding_value)
    return resized,(x1,y1,x2,y2),metadata

def crop_face(image: np.ndarray, box: tuple[int,int,int,int], margin: int, size: tuple[int,int]) -> np.ndarray:
    resized,_,_=crop_and_resize_face(image,box,margin/max(1,box[2]),size)
    return resized
