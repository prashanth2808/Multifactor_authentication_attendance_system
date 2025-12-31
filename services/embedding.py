# services/embedding.py
"""
Face embedding using ArcFace model
"""

import numpy as np
import cv2
from pathlib import Path
from rich.console import Console
from services.arcface import get_arcface_embedding
from services.face_detection import get_cropped_face

console = Console()

def get_face_embedding(face_img: np.ndarray) -> np.ndarray | None:
    """
    Generate face embedding using ORIGINAL ArcFace model (OFFICIAL SOURCES)
    Uses the cropped face image directly with original ArcFace ONNX
    Returns 512-dimensional ORIGINAL ArcFace vector
    """
    try:
        if face_img is None or face_img.size == 0:
            return None
        
        # Method 1: Use the provided cropped face directly
        if face_img.shape[:2] == (112, 112):
            console.print("[cyan]Using cropped face (112x112) for ArcFace[/cyan]")
            embedding = get_arcface_embedding(face_img)
            if embedding is not None:
                return embedding
        
        # Method 2: Try to get face from original frame
        try:
            from utils.camera import _last_full_frame
            if hasattr(_last_full_frame, '__len__') and len(_last_full_frame) > 0:
                console.print("[cyan]Detecting face from original frame for ArcFace[/cyan]")
                cropped_face = get_cropped_face(_last_full_frame)
                if cropped_face is not None:
                    embedding = get_arcface_embedding(cropped_face)
                    if embedding is not None:
                        return embedding
        except Exception as e:
            console.print(f"[yellow]Could not use original frame: {e}[/yellow]")
        
        # Method 3: Resize the provided face to 112x112
        console.print("[cyan]Resizing face to 112x112 for ArcFace[/cyan]")
        face_resized = cv2.resize(face_img, (112, 112))
        embedding = get_arcface_embedding(face_resized)
        
        return embedding
        
    except Exception as e:
        console.print(f"[red]Failed to generate ArcFace embedding: {e}[/red]")
        return None
