# utils/image.py
"""
Image preprocessing for ArcFace ONNX model
Input: BGR image (112x112, numpy)
Output: float32 blob (1, 3, 112, 112) or None
"""

import numpy as np
import cv2
from rich.console import Console

console = Console()

# ArcFace input mean/std (from InsightFace)
MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
STD = np.array([128.0, 128.0, 128.0], dtype=np.float32)

def preprocess_image(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Preprocess BGR image for ArcFace:
    - Convert BGR to RGB
    - Subtract mean, divide by std
    - HWC → CHW
    - Add batch dim
    Returns: (1, 3, 112, 112) float32
    """
    try:
        if img_bgr.shape != (112, 112, 3):
            console.print(f"[red]Invalid image shape: {img_bgr.shape}, expected (112,112,3)[/red]")
            return None

        # BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Convert to float32
        img = img_rgb.astype(np.float32)

        # Normalize: (img - mean) / std
        img = (img - MEAN) / STD

        # HWC → CHW
        img = img.transpose(2, 0, 1)  # (3, 112, 112)

        # Add batch dimension
        blob = np.expand_dims(img, axis=0)  # (1, 3, 112, 112)

        return blob

    except Exception as e:
        console.print(f"[red]Preprocessing failed: {e}[/red]")
        return None