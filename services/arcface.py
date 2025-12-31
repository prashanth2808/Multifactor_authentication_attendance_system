# services/arcface.py
"""
ArcFace model implementation using direct ONNX download
Downloads and uses ArcFace models from official sources
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
import requests
from rich.console import Console
from typing import Optional
import os
import zipfile

console = Console()

# Global ArcFace model
_arcface_session = None
_model_path = None

# Buffalo_L model - Standard InsightFace model package
ARCFACE_MODELS = {
    "buffalo_l": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "filename": "buffalo_l.zip",
        "description": "InsightFace Buffalo_L"
    }
}

def download_arcface_model(model_type="buffalo_l"):
    """Download and extract ArcFace model from buffalo_l package"""
    # Get absolute path to models directory
    script_dir = Path(__file__).parent.parent
    models_dir = script_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    if model_type not in ARCFACE_MODELS:
        console.print(f"[red]Unknown model type: {model_type}. Available: {list(ARCFACE_MODELS.keys())}[/red]")
        return None
        
    # Check if ArcFace model already exists (extracted)
    arcface_model_path = models_dir / "w600k_r50.onnx"
    if arcface_model_path.exists():
        console.print(f"[green]✓ ArcFace model already exists: {arcface_model_path}[/green]")
        return str(arcface_model_path)
    
    model_info = ARCFACE_MODELS[model_type]
    zip_path = models_dir / model_info["filename"]
    
    # Download ZIP if not exists
    if not zip_path.exists():
        console.print(f"[yellow]Downloading {model_info['description']}...[/yellow]")
        console.print(f"[cyan]Target: {zip_path}[/cyan]")
        
        try:
            download_result = download_direct(model_info["url"], zip_path)
            if download_result is None:
                return None
        except Exception as e:
            console.print(f"[red]Failed to download model: {e}[/red]")
            if zip_path.exists():
                zip_path.unlink()
            return None
    
    # Extract ArcFace model from ZIP
    try:
        console.print(f"[yellow]Extracting ArcFace model from {zip_path}...[/yellow]")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # List contents
            file_list = zip_ref.namelist()
            console.print(f"[cyan]ZIP contents: {file_list}[/cyan]")
            
            # Look for ArcFace ONNX model
            arcface_file = None
            for file in file_list:
                if 'w600k_r50' in file and file.endswith('.onnx'):
                    arcface_file = file
                    break
            
            if arcface_file is None:
                console.print(f"[red]No ArcFace model found in ZIP. Available files: {file_list}[/red]")
                return None
            
            # Extract the ArcFace model
            zip_ref.extract(arcface_file, models_dir)
            extracted_path = models_dir / arcface_file
            
            # Rename to standard name
            if extracted_path != arcface_model_path:
                extracted_path.rename(arcface_model_path)
            
            console.print(f"[bold green]✓ ArcFace model extracted: {arcface_model_path}[/bold green]")
            return str(arcface_model_path)
            
    except Exception as e:
        console.print(f"[red]Failed to extract ArcFace model: {e}[/red]")
        return None

def download_direct(url, model_path):
    """Download from direct URL"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    with open(model_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    console.print(f"\r[cyan]Progress: {progress:.1f}%[/cyan]", end="")
    
    console.print(f"\n[bold green]✓ ORIGINAL ArcFace model downloaded: {model_path}[/bold green]")
    return str(model_path)

def download_from_gdrive(gdrive_url, model_path):
    """Download from Google Drive with gdown"""
    try:
        import gdown
        
        # Extract file ID from Google Drive URL
        if "uc?id=" in gdrive_url:
            file_id = gdrive_url.split("uc?id=")[1]
        else:
            console.print("[red]Invalid Google Drive URL format[/red]")
            return None
            
        # Download using gdown
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(model_path), quiet=False)
        
        if model_path.exists() and model_path.stat().st_size > 0:
            console.print(f"\n[bold green]✓ ArcFace model downloaded: {model_path}[/bold green]")
            return str(model_path)
        else:
            console.print("[red]Download failed or file is empty[/red]")
            return None
            
    except ImportError:
        console.print("[red]gdown not installed. Installing...[/red]")
        os.system("pip install gdown")
        return download_from_gdrive(gdrive_url, model_path)
    except Exception as e:
        console.print(f"[red]Google Drive download failed: {e}[/red]")
        return None

def _get_arcface_session():
    """Get ArcFace ONNX session"""
    global _arcface_session, _model_path
    
    if _arcface_session is None:
        # Download model if needed
        _model_path = download_arcface_model("buffalo_l")  # Use buffalo_l model
            
        if _model_path is None:
            return None
        
        try:
            # Load ONNX model
            _arcface_session = ort.InferenceSession(
                _model_path,
                providers=['CPUExecutionProvider']
            )
            console.print("[bold green]✓ ArcFace ONNX session created[/bold green]")
            
            # Print model info
            input_shape = _arcface_session.get_inputs()[0].shape
            output_shape = _arcface_session.get_outputs()[0].shape
            console.print(f"[cyan]Input shape: {input_shape}[/cyan]")
            console.print(f"[cyan]Output shape: {output_shape}[/cyan]")
            
        except Exception as e:
            console.print(f"[red]Failed to load ArcFace ONNX: {e}[/red]")
            return None
    
    return _arcface_session

def preprocess_face_for_arcface(face_img: np.ndarray) -> np.ndarray:
    """
    Preprocess face image for ArcFace model
    Input: (112, 112, 3) BGR image
    Output: (1, 3, 112, 112) float32 tensor
    """
    if face_img.shape != (112, 112, 3):
        face_img = cv2.resize(face_img, (112, 112))
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] then to [-1, 1] (standard ArcFace preprocessing)
    face_normalized = face_rgb.astype(np.float32) / 255.0
    face_normalized = (face_normalized - 0.5) / 0.5
    
    # HWC to CHW
    face_chw = face_normalized.transpose(2, 0, 1)
    
    # Add batch dimension
    face_batch = np.expand_dims(face_chw, axis=0)
    
    return face_batch

def get_arcface_embedding(face_img: np.ndarray) -> Optional[np.ndarray]:
    """
    Generate embedding using ArcFace model
    Input: (112, 112, 3) BGR face image
    Output: 512-dimensional embedding vector
    """
    try:
        session = _get_arcface_session()
        if session is None:
            return None
        
        # Preprocess image
        input_blob = preprocess_face_for_arcface(face_img)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_blob})
        
        # Get embedding (first output)
        embedding = outputs[0][0]  # Remove batch dimension
        
        # Normalize embedding (L2 normalization)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        console.print("[bold green]✓ ArcFace embedding generated (512-dim)[/bold green]")
        return embedding.astype(np.float32)
        
    except Exception as e:
        console.print(f"[red]ArcFace embedding failed: {e}[/red]")
        return None

def verify_arcface_model():
    """Verify the ArcFace model is working"""
    console.print("[yellow]Verifying ArcFace model...[/yellow]")
    
    # Create test image
    test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # Get embedding
    embedding = get_arcface_embedding(test_img)
    
    if embedding is not None:
        console.print(f"[green]✓ Model verification successful[/green]")
        console.print(f"[cyan]Embedding shape: {embedding.shape}[/cyan]")
        console.print(f"[cyan]Embedding norm: {np.linalg.norm(embedding):.6f}[/cyan]")
        return True
    else:
        console.print("[red]✗ Model verification failed[/red]")
        return False