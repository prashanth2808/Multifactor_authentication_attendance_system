# services/voice_embedding_local.py
# ECAPA-TDNN using locally downloaded model (NO SYMLINKS)

import warnings
warnings.filterwarnings("ignore")

# === Optional torchaudio patch ===
# Some environments (notably certain Windows builds) expose a torchaudio variant
# missing set_audio_backend; keep the old behavior but do not fail import on Linux/cloud.
try:
    import torchaudio  # type: ignore
    if not hasattr(torchaudio, "set_audio_backend"):
        def dummy_backend(backend):
            pass
        torchaudio.set_audio_backend = dummy_backend  # type: ignore
except Exception:  # pragma: no cover
    torchaudio = None  # type: ignore

import os
import numpy as np
import torch
# sounddevice is only needed for local CLI recording.
# In cloud deployment, audio is uploaded from the browser.
try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover
    sd = None  # type: ignore
import logging
from rich.console import Console
import time
import wave
import datetime

console = Console()
logger = logging.getLogger(__name__)

# Global variables
_verification_model = None
_vad_model = None

def _load_ecapa_model():
    """Load ECAPA-TDNN model from local files"""
    global _verification_model
    
    if _verification_model is not None:
        return _verification_model
    
    try:
        console.print("[cyan]Loading ECAPA-TDNN from local files...[/cyan]")
        
        from speechbrain.inference import SpeakerRecognition
        
        # Use the locally downloaded model
        local_model_path = "pretrained_models/spkrec-ecapa-voxceleb"
        
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Local model not found at {local_model_path}. Run download_ecapa_manual.py first!")
        
        # Load ECAPA-TDNN model from local path
        _verification_model = SpeakerRecognition.from_hparams(
            source=local_model_path,
            savedir=local_model_path,
            run_opts={"device": "cpu"}
        )
        
        console.print("[bold green]✅ ECAPA-TDNN loaded from local files![/bold green]")
        return _verification_model
        
    except Exception as e:
        console.print(f"[red]❌ ECAPA-TDNN loading failed: {e}[/red]")
        raise e

def _load_vad_model():
    """Load Silero VAD model"""
    global _vad_model
    
    if _vad_model is None:
        try:
            console.print("[cyan]Loading Silero VAD model...[/cyan]")
            _vad_model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            console.print("[green]✅ Silero VAD loaded successfully![/green]")
        except Exception as e:
            console.print(f"[red]❌ Silero VAD loading failed: {e}[/red]")
            raise e
    
    return _vad_model

def record_audio(duration=5.0, fs=16000):
    """Record audio from microphone (local CLI use only).

    In cloud/web deployments this must NOT be used. The browser records audio and uploads it.
    """
    if sd is None:
        raise RuntimeError("sounddevice is not available. Use browser upload + embed from waveform instead.")

    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()

def apply_vad(audio_np, fs=16000):
    """Apply Silero VAD for speech detection (Silero VAD only)"""
    vad_model = _load_vad_model()
    
    # Normalize audio
    audio_float = audio_np.astype(np.float32)
    if np.max(np.abs(audio_float)) > 1.0:
        audio_float /= np.max(np.abs(audio_float))
    
    # Convert to torch tensor
    audio_tensor = torch.tensor(audio_float)
    
    # Silero VAD expects exactly 512 samples for 16kHz
    speech_timestamps = []
    chunk_size = 512  # 32ms chunks for 16kHz
    
    for i in range(0, len(audio_tensor) - chunk_size, chunk_size):
        chunk = audio_tensor[i:i + chunk_size]
        
        # VAD model expects (tensor, sample_rate) - chunk must be exactly 512 samples
        speech_prob = vad_model(chunk, fs).item()
        
        if speech_prob > 0.5:  # Speech detected
            speech_timestamps.append({
                'start': i,
                'end': i + chunk_size
            })
    
    if not speech_timestamps:
        console.print("[red]⚠️  Silero VAD found no speech in audio[/red]")
        return np.array([])  # Return empty array instead of fallback
    
    # Extract speech segments
    speech_parts = []
    for ts in speech_timestamps:
        speech_parts.append(audio_float[ts['start']:ts['end']])
    
    speech_audio = np.concatenate(speech_parts) if speech_parts else np.array([])
    
    if len(speech_audio) < 8000:  # Less than 0.5 seconds
        console.print("[red]⚠️  Silero VAD detected insufficient speech (< 0.5 seconds)[/red]")
        return np.array([])  # Return empty array instead of fallback
    
    console.print(f"[green]✅ Silero VAD detected {len(speech_audio)} speech samples[/green]")
    return speech_audio


def normalize_loudness(audio, target_rms=0.05):
    """Normalize audio loudness"""
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-8:
        return audio
    gain = target_rms / rms
    gain = min(gain, 10.0)  # max 20dB boost
    return np.clip(audio * gain, -1.0, 1.0)

def get_ecapa_embedding(audio_np):
    """Generate ECAPA-TDNN embedding from audio"""
    try:
        verification_model = _load_ecapa_model()
        
        # Ensure audio is proper format
        audio = audio_np.astype(np.float32)
        if len(audio) == 0:
            console.print("[yellow]⚠️  Empty audio for embedding[/yellow]")
            return None
        
        # Normalize loudness
        audio = normalize_loudness(audio)
        
        # Convert to torch tensor
        audio_tensor = torch.tensor(audio).unsqueeze(0)  # [1, T]
        
        # Generate embedding
        with torch.no_grad():
            embedding = verification_model.encode_batch(audio_tensor)
            embedding = embedding.squeeze(0).cpu().numpy()
        
        console.print(f"[green]✅ ECAPA-TDNN embedding generated: {embedding.shape}[/green]")
        return embedding
        
    except Exception as e:
        console.print(f"[red]❌ ECAPA-TDNN embedding failed: {e}[/red]")
        raise e

def save_audio_backups(audio_clips, user_id):
    """Save 3 audio clips as backup WAV files for future model changes"""
    try:
        # Create backup directory
        backup_dir = "voice_backups"
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
            console.print(f"[cyan]📁 Created backup directory: {backup_dir}[/cyan]")
        
        # Create user-specific directory
        user_backup_dir = os.path.join(backup_dir, f"user_{user_id}")
        if not os.path.exists(user_backup_dir):
            os.makedirs(user_backup_dir)
        
        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        audio_backup_paths = []
        
        for i, audio_clip in enumerate(audio_clips, 1):
            # Normalize audio to 16-bit integers
            audio_normalized = (audio_clip * 32767).astype(np.int16)
            
            # Create filename
            filename = f"{user_id}_clip_{i}_{timestamp}.wav"
            filepath = os.path.join(user_backup_dir, filename)
            
            # Save as WAV file
            with wave.open(filepath, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16kHz
                wav_file.writeframes(audio_normalized.tobytes())
            
            audio_backup_paths.append(filepath)
            console.print(f"[green]💾 Saved backup audio: {filename}[/green]")
        
        console.print(f"[bold green]✅ All 3 audio backups saved for user {user_id}[/bold green]")
        return audio_backup_paths
        
    except Exception as e:
        console.print(f"[red]❌ Failed to save audio backups: {e}[/red]")
        return []

def record_and_embed_three_times(duration_per_clip=7.0, user_id=None):
    """Record 3 voice clips, generate averaged embedding, and save audio backups"""
    console.print("[bold cyan]🎤 ECAPA-TDNN Voice Registration — 3 Clips[/bold cyan]")
    
    embeddings = []
    audio_clips = []  # Store all successful audio clips
    
    for i in range(3):
        console.print(f"\\n[bold yellow]📹 Clip {i+1}/3[/bold yellow]")
        console.print("[cyan]Speak clearly and loudly...[/cyan]")
        console.print("[cyan]Say: 'Hello, my name is [your name]'[/cyan]")
        
        # Countdown
        for j in range(3, 0, -1):
            console.print(f"[yellow]Starting in {j}...[/yellow]")
            time.sleep(1)
        
        console.print("[bold red]🔴 RECORDING NOW...[/bold red]")
        
        # Record audio
        raw_audio = record_audio(duration_per_clip)
        
        # Apply VAD
        cleaned = apply_vad(raw_audio)
        if len(cleaned) == 0:
            console.print("[red]❌ No speech detected by Silero VAD - speak louder! Retrying...[/red]")
            i -= 1
            continue
        if len(cleaned) < 16000:  # less than 1 second speech
            console.print("[red]❌ Not enough speech detected - speak louder! Retrying...[/red]")
            i -= 1
            continue
        
        # Generate embedding
        emb = get_ecapa_embedding(cleaned)
        if emb is None:
            console.print("[red]❌ Failed to generate embedding - retrying...[/red]")
            i -= 1
            continue
        
        norm = np.linalg.norm(emb)
        console.print(f"[green]✅ Clip {i+1} successful - embedding norm: {norm:.2f}[/green]")
        embeddings.append(emb)
        audio_clips.append(cleaned)  # Store the cleaned audio clip
        time.sleep(1)
    
    # Average the embeddings
    final_embedding = np.mean(embeddings, axis=0)
    console.print("[bold green]🎉 VOICE REGISTRATION COMPLETE — ECAPA-TDNN[/bold green]")
    console.print(f"[cyan]Final embedding shape: {final_embedding.shape}[/cyan]")
    
    # Save audio backups if user_id provided
    audio_backup_paths = []
    if user_id:
        audio_backup_paths = save_audio_backups(audio_clips, user_id)
    
    # Return the best audio clip (longest speech duration)
    best_audio_clip = max(audio_clips, key=len) if audio_clips else audio_clips[0]
    
    return final_embedding, best_audio_clip, audio_backup_paths

def verify_voice_live(stored_emb, duration=5.0, threshold=0.72):
    """Verify voice using ECAPA-TDNN"""
    console.print("[bold blue]🎤 Voice Verification — ECAPA-TDNN[/bold blue]")
    console.print("[cyan]Speak the same phrase clearly...[/cyan]")
    
    # Record live audio
    raw_audio = record_audio(duration)
    
    # Apply VAD
    cleaned = apply_vad(raw_audio)
    if len(cleaned) == 0:
        console.print("[red]❌ No speech detected by Silero VAD - speak louder![/red]")
        return 0.0, False
    if len(cleaned) < 8000:
        console.print("[red]❌ Insufficient speech detected - speak louder![/red]")
        return 0.0, False
    
    # Generate live embedding
    live_emb = get_ecapa_embedding(cleaned)
    if live_emb is None:
        console.print("[red]❌ Failed to generate embedding[/red]")
        return 0.0, False
    
    # Ensure both embeddings are 1D arrays
    stored_emb_flat = np.array(stored_emb).flatten()
    live_emb_flat = live_emb.flatten()
    
    # Cosine similarity
    score = np.dot(stored_emb_flat, live_emb_flat) / (
        np.linalg.norm(stored_emb_flat) * np.linalg.norm(live_emb_flat) + 1e-8
    )
    
    passed = score >= threshold
    status = "✅ PASS" if passed else "❌ FAIL"
    console.print(f"[bold]ECAPA-TDNN Score: {score:.1%} → {status}[/bold]")
    
    return score, passed

# Aliases for backward compatibility
get_voice_embedding = get_ecapa_embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity helper for voice embeddings."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))

def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resampler to avoid extra deps in hosted mode."""
    if orig_sr == target_sr:
        return audio
    ratio = orig_sr / target_sr
    new_len = int(len(audio) / ratio)
    if new_len <= 1:
        return audio
    x = np.arange(len(audio), dtype=np.float32)
    xp = np.linspace(0, len(audio) - 1, num=new_len, dtype=np.float32)
    return np.interp(xp, x, audio).astype(np.float32)


def get_voice_embedding_from_audio_bytes(audio_bytes: bytes, expected_sr: int = 16000) -> np.ndarray | None:
    """Cloud-safe helper: decode uploaded WAV bytes and extract an ECAPA embedding.

    Mirrors CLI behavior:
    - normalize loudness BEFORE VAD (helps VAD on quiet mics)
    - require minimum speech length after VAD
    """
    try:
        from services.io_helpers import decode_wav_bytes_to_float32

        decoded = decode_wav_bytes_to_float32(audio_bytes)
        if decoded is None:
            return None
        audio, sr = decoded
        if sr != expected_sr:
            audio = _resample_linear(audio, sr, expected_sr)
            sr = expected_sr

        # Pre-VAD loudness normalization helps Silero VAD detect speech on quiet inputs
        audio = normalize_loudness(np.asarray(audio, dtype=np.float32))

        cleaned = apply_vad(audio, fs=sr)
        if len(cleaned) == 0:
            return None
        # CLI checks for minimum speech; keep the same idea here
        if len(cleaned) < 8000:  # < 0.5s at 16kHz
            return None

        return get_ecapa_embedding(cleaned)
    except Exception as e:
        console.print(f"[red]Failed to embed from audio bytes: {e}[/red]")
        return None


def verify_voice_from_audio_bytes(stored_emb, audio_bytes: bytes, threshold: float = 0.72) -> tuple[float, bool]:
    """Cloud-safe voice verification from uploaded audio bytes."""
    score, passed, _ = verify_voice_from_audio_bytes_detailed(stored_emb, audio_bytes, threshold=threshold)
    return score, passed


def verify_voice_from_audio_bytes_detailed(
    stored_emb,
    audio_bytes: bytes,
    threshold: float = 0.72,
) -> tuple[float, bool, str]:
    """Like CLI: return a reason code for failures.

    reason:
      - ok
      - no_speech
      - embedding_failed
      - below_threshold
    """
    # Debug: basic bytes size
    try:
        console.print(f"[dim]Web voice bytes: {len(audio_bytes)} bytes | threshold={threshold:.3f}[/dim]")
    except Exception:
        pass

    live_emb = get_voice_embedding_from_audio_bytes(audio_bytes)
    if live_emb is None:
        try:
            console.print("[yellow]Web voice: embedding not generated (no_speech / decode issue)[/yellow]")
        except Exception:
            pass
        return 0.0, False, "no_speech"

    stored_emb_flat = np.array(stored_emb, dtype=np.float32).flatten()
    live_emb_flat = np.array(live_emb, dtype=np.float32).flatten()
    score = float(np.dot(stored_emb_flat, live_emb_flat) / (
        np.linalg.norm(stored_emb_flat) * np.linalg.norm(live_emb_flat) + 1e-8
    ))
    passed = bool(score >= threshold)

    # Print same style as CLI for easy comparison
    try:
        status = "✅ PASS" if passed else "❌ FAIL"
        console.print(f"[bold]ECAPA-TDNN Score (web): {score:.1%} → {status}[/bold]")
    except Exception:
        pass

    return score, passed, ("ok" if passed else "below_threshold")


# NOTE: Do not auto-initialize models on import.
# In API/server contexts this module is imported during startup; eager loading can slow boot.
# Models will be loaded lazily on first use via _load_ecapa_model() / _load_vad_model().