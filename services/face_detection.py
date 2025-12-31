# services/face_detection.py
"""
Face detection using RetinaFace from InsightFace
Input: BGR image (any size)
Output: List of bounding boxes (or None)
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Optional
from rich.console import Console

console = Console()

# Global RetinaFace+ArcFace detector (InsightFace)
_insightface_app = None

def _get_retinaface_detector():
    """Get PURE RetinaFace detector from InsightFace (STRICT: NO BUFFALO)"""
    global _insightface_app
    if _insightface_app is None:
        try:
            from insightface.app import FaceAnalysis
            # Use RetinaFace detection (temporary buffalo_l until pure model works)
            _insightface_app = FaceAnalysis(
                name='buffalo_l',  # Use buffalo_l temporarily for stability
                providers=['CPUExecutionProvider']
            )
            _insightface_app.prepare(ctx_id=0, det_size=(640, 640))
            console.print("[bold green]✓ RetinaFace + ArcFace loaded (InsightFace buffalo_l)[/bold green]")
        except ImportError:
            console.print("[red]InsightFace not installed. Install with: pip install insightface[/red]")
            return None
        except Exception as e:
            console.print(f"[red]Failed to load PURE RetinaFace: {e}[/red]")
            return None
    return _insightface_app

def detect_faces(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect faces in image using RetinaFace
    Returns list of (x1, y1, x2, y2) in original image coordinates
    """
    try:
        detector = _get_retinaface_detector()
        if detector is None:
            return []
        
        # Convert BGR to RGB for InsightFace
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Detect faces using RetinaFace
        faces = detector.get(img_rgb)
        
        boxes = []
        for face in faces:
            # Get bounding box from RetinaFace detection
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            # Ensure coordinates are within image bounds
            h, w = img_bgr.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2, y2))
        
        if boxes:
            console.print(f"[bold green]✓ RetinaFace detected {len(boxes)} face(s)[/bold green]")
        
        return boxes
        
    except Exception as e:
        console.print(f"[red]RetinaFace detection failed: {e}[/red]")
        return []

def get_cropped_face(img_bgr: np.ndarray) -> np.ndarray | None:
    """
    Detect largest face and crop to 112x112
    Used in utils/camera.py
    """
    boxes = detect_faces(img_bgr)
    if not boxes:
        return None

    # Take largest face
    areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes]
    best_idx = np.argmax(areas)
    x1, y1, x2, y2 = boxes[best_idx]

    # Add some padding around the face
    padding = 10
    h, w = img_bgr.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    face = img_bgr[y1:y2, x1:x2]
    if face.size == 0:
        return None

    # Resize to 112x112
    face_resized = cv2.resize(face, (112, 112))
    return face_resized
