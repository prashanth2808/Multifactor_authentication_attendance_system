# utils/camera.py
"""
Capture face image using OpenCV webcam
Uses RetinaFace for accurate detection (fallback to Haar)
Returns: 112x112 BGR face image or None
"""

import cv2
import numpy as np
from rich.console import Console
from services.face_detection import get_cropped_face

console = Console()

# STRICTLY RetinaFace detection ONLY - NO HAAR, NO MTCNN, NO OTHER SHITS
# Global variable to store the last frame for ArcFace embedding
_last_full_frame = None

def capture_face_image() -> np.ndarray | None:
    """
    Open webcam, show live feed with face rectangle
    Press SPACE to capture, ESC to cancel
    Returns cropped face image (112x112) or None
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        console.print("[red]Error: Could not open webcam[/red]")
        return None

    console.print("[bold cyan]Webcam opened. Look straight. Press SPACE to capture.[/bold cyan]")
    
    captured = False
    face_img = None

    while not captured:
        ret, frame = cap.read()
        if not ret:
            console.print("[red]Failed to grab frame[/red]")
            break
            
        # Store the full frame globally for ArcFace embedding
        global _last_full_frame
        _last_full_frame = frame.copy()

        # STRICTLY use RetinaFace only - NO OTHER DETECTION METHODS
        from services.face_detection import detect_faces
        face_boxes = detect_faces(frame)
        
        # Convert to (x, y, w, h) format for display
        faces = []
        for x1, y1, x2, y2 in face_boxes:
            faces.append((x1, y1, x2 - x1, y2 - y1))
        
        # Get cropped face for capture
        cropped = get_cropped_face(frame)

        # Draw rectangles - STRICTLY RetinaFace detection
        for (x, y, w, h) in faces:
            color = (0, 255, 0)  # Always green for RetinaFace
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, "RETINAFACE + ARCFACE", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Status text
        status = "Press SPACE to capture" if len(faces) > 0 else "No face detected"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Face Capture - SPACE to capture, ESC to cancel', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            if len(faces) == 1 and cropped is not None:
                face_img = cropped
                captured = True
                console.print("[bold green]✓ Face captured with RetinaFace + ArcFace![/bold green]")
            elif len(faces) > 1:
                console.print("[yellow]Multiple faces detected. Show only one face.[/yellow]")
            else:
                console.print("[yellow]No face detected. Try again.[/yellow]")

        elif key == 27:  # ESC
            console.print("[bold red]Capture cancelled by user[/bold red]")
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return face_img

def capture_face_burst() -> list[np.ndarray] | None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        console.print("[red]Error: Could not open webcam[/red]")
        return None

    console.print("[bold cyan]Look at camera → Press SPACE once to start[/bold cyan]")
    face_images = []
    started = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        global _last_full_frame
        _last_full_frame = frame.copy()

        from services.face_detection import detect_faces
        face_boxes = detect_faces(frame)
        faces = [(x1, y1, x2-x1, y2-y1) for x1, y1, x2, y2 in face_boxes]

        # Draw face box (thin green)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # TOP-LEFT TEXT ONLY — small and clean
        y_pos = 35
        def add_text(text, color=(200, 200, 200)):
            nonlocal y_pos
            cv2.putText(frame, text, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_pos += 28

        if not started:
            if len(faces) == 1:
                add_text("Ready — Press SPACE to start", (0, 255, 255))
            else:
                add_text("Show only one face", (0, 255, 255))
        else:
            add_text("Capturing in progress...", (0, 255, 255))

        cv2.imshow('Face Registration', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32 and not started and len(faces) == 1:  # SPACE
            started = True
            console.print("[bold yellow]Starting capture in 5 seconds...[/bold yellow]")

            # ONE clean 5-second countdown (small, top-left)
            for sec in range(5, 0, -1):
                ret_c, frame_c = cap.read()
                if ret_c:
                    _last_full_frame = frame_c.copy()
                    y = 50
                    cv2.putText(frame_c, f"Capturing in {sec}...", (15, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)
                    cv2.imshow('Face Registration', frame_c)
                cv2.waitKey(1000)

            # Take 3 photos instantly
            for i in range(1, 4):
                ret_p, frame_p = cap.read()
                if not ret_p:
                    continue
                _last_full_frame = frame_p.copy()
                cropped = get_cropped_face(frame_p)
                if cropped is not None:
                    face_images.append(cropped)
                    console.print(f"[bold green]Photo {i}/3 captured[/bold green]")
                    # Small feedback
                    cv2.putText(frame_p, f"Photo {i}/3", (15, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow('Face Registration', frame_p)
                cv2.waitKey(400)

            if len(face_images) == 3:
                console.print("[bold green]All 3 photos captured successfully![/bold green]")
                break
            else:
                console.print("[red]Failed. Try again.[/red]")
                face_images.clear()
                started = False

    cap.release()
    cv2.destroyAllWindows()
    return face_images if len(face_images) == 3 else None