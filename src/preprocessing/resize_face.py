"""Aspect-ratio-preserving resize for detected face crops.

This module must only receive a validated face crop. Raw source images should be
passed to the configured detector at their original resolution before this step.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ResizeMetadata:
    """Geometry used to create a fixed-size FaceNet input image."""

    source_width: int
    source_height: int
    resized_width: int
    resized_height: int
    padding_left: int
    padding_top: int
    padding_right: int
    padding_bottom: int
    scale: float


def resize_face_crop(
    face_crop: np.ndarray,
    output_size: tuple[int, int] = (160, 160),
    padding_value: tuple[int, int, int] = (0, 0, 0),
) -> tuple[np.ndarray, ResizeMetadata]:
    """Resize a BGR face crop without distortion and center-pad the remainder.

    Args:
        face_crop: Non-empty OpenCV image with one or three channels.
        output_size: Target ``(width, height)``. FaceNet uses ``(160, 160)``.
        padding_value: BGR value used for any letterbox padding.

    Returns:
        The fixed-size image and reproducible resize geometry.

    Raises:
        ValueError: If the input image or requested output size is invalid.
    """
    if not isinstance(face_crop, np.ndarray) or face_crop.size == 0:
        raise ValueError("Face crop must be a non-empty NumPy array")
    if face_crop.ndim not in (2, 3):
        raise ValueError(f"Unsupported face crop shape: {face_crop.shape}")
    if face_crop.ndim == 3 and face_crop.shape[2] not in (1, 3):
        raise ValueError(f"Unsupported channel count: {face_crop.shape[2]}")

    target_width, target_height = map(int, output_size)
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"Invalid output size: {output_size}")

    source_height, source_width = face_crop.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid face crop dimensions: {face_crop.shape}")

    scale = min(target_width / source_width, target_height / source_height)
    resized_width = min(target_width, max(1, round(source_width * scale)))
    resized_height = min(target_height, max(1, round(source_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(
        face_crop,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    horizontal = target_width - resized_width
    vertical = target_height - resized_height
    left, right = horizontal // 2, horizontal - horizontal // 2
    top, bottom = vertical // 2, vertical - vertical // 2
    border_value: int | tuple[int, int, int]
    border_value = 0 if face_crop.ndim == 2 else tuple(map(int, padding_value))
    output = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=border_value,
    )
    if output.shape[:2] != (target_height, target_width):
        raise RuntimeError(f"Resize produced unexpected shape: {output.shape}")

    metadata = ResizeMetadata(
        source_width=source_width,
        source_height=source_height,
        resized_width=resized_width,
        resized_height=resized_height,
        padding_left=left,
        padding_top=top,
        padding_right=right,
        padding_bottom=bottom,
        scale=scale,
    )
    return output, metadata
