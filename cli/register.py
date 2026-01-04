# cli/register.py
"""
Register a new user with 3 face photos + 3 voice clips + SAVES VOICE .WAV FILE
Command: python main.py register --name "Name" --email "email@x.com"
"""

import typer
import os
from datetime import datetime
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from utils.camera import capture_face_burst
from services.embedding import get_face_embedding
from services.voice_embedding import record_and_embed_three_times  # ← Updated path
from db.user_repo import save_user, find_user_by_email

console = Console()

# Create folder for voice recordings
VOICE_DIR = "captured_voices"
os.makedirs(VOICE_DIR, exist_ok=True)

def register(
    name: str = typer.Option(..., "--name", "-n", help="Full name of the user"),
    email: str = typer.Option(..., "--email", "-e", help="Email address")
):
    """
    Capture 3 face photos + 3 voice clips, save .wav file, generate embeddings, save to Supabase Postgres
    """
    console.print(f"[bold blue]Starting registration for:[/bold blue] {name} ({email})")
    
    # Check if user already exists
    existing = find_user_by_email(email)
    if existing:
        console.print(f"[bold red]User with email {email} already registered[/bold red]")
        raise typer.Exit(code=1)

    # === FACE CAPTURE ===
    face_embeddings = []
    console.print("[yellow]Capturing 3 face photos...[/yellow]")
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), transient=True) as progress:
        task = progress.add_task("Preparing camera...", total=None)
        face_images = capture_face_burst()
        
        if face_images is None or len(face_images) != 3:
            console.print("[bold red]Failed to capture 3 valid face photos[/bold red]")
            raise typer.Exit(code=1)

        progress.update(task, description="Processing face embeddings...", total=3)
        for i, img in enumerate(face_images):
            embedding = get_face_embedding(img)
            if embedding is None:
                console.print(f"[red]Failed to generate embedding for photo {i+1}[/red]")
                raise typer.Exit(code=1)
            face_embeddings.append(embedding.tolist())
            progress.advance(task)
            console.print(f"[green]Embedding {i+1}/3 generated[/green]")

    console.print("[bold green]All 3 face photos captured and processed![/bold green]")

    # === VOICE CAPTURE + SAVE .WAV FILE ===
    console.print("\n[yellow]Capturing 3 voice clips + saving .wav file...[/yellow]")
    console.print(f"[cyan]Say clearly: \"Hello, My name is {name}\"[/cyan]")

    try:
        # Create a readable + unique backup folder id
        # Example: "Prashanth_S_d4c74594" (name + short hash of email)
        safe_name = "".join(c if c.isalnum() or c in " _-" else "_" for c in name).strip()
        safe_name = safe_name.replace(" ", "_")

        import hashlib
        email_hash = hashlib.md5(email.encode()).hexdigest()[:8]
        backup_folder_id = f"{safe_name}_{email_hash}" if safe_name else email_hash

        # This returns embedding, best audio clip, and backup paths
        # Backups will be saved under: voice_backups/user_<backup_folder_id>/
        voice_embedding, best_audio_clip, audio_backup_paths = record_and_embed_three_times(
            duration_per_clip=7.0,
            user_id=backup_folder_id
        )
        voice_embedding_list = voice_embedding.tolist()

        # Save the best voice clip as .wav (legacy format)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        voice_filename = f"{safe_name}_{timestamp}_voice.wav" if safe_name else f"{email_hash}_{timestamp}_voice.wav"
        voice_path = os.path.join(VOICE_DIR, voice_filename)

        try:
            import soundfile as sf  # local-only dependency
        except Exception as e:
            console.print(f"[bold red]Missing dependency 'soundfile'. Install local extras: pip install -r requirements_local.txt ({e})[/bold red]")
            raise

        sf.write(voice_path, best_audio_clip, samplerate=16000)
        console.print(f"[bold green]Voice recording saved: {voice_path}[/bold green]")
        
        # Display backup information
        if audio_backup_paths:
            console.print(f"[bold cyan]🔒 Audio backups saved ({len(audio_backup_paths)} files)[/bold cyan]")

    except Exception as e:
        console.print(f"[bold red]Voice enrollment failed: {e}[/bold red]")
        raise typer.Exit(code=1)

    # === SAVE TO DATABASE (Supabase Postgres) ===
    user_data = {
        "name": name,
        "email": email,
        "face_embeddings": face_embeddings,
        "voice_embedding": voice_embedding_list,
        "voice_backup_paths": audio_backup_paths,  # NEW: all 3 clips for future model changes
        "backup_user_id": backup_folder_id,   # NEW: backup folder identifier (readable + unique)
        "photo_count": 3,
        "voice_clips": 3,
        "registered_at": datetime.utcnow()
    }

    inserted = save_user(user_data)
    if inserted and inserted.get("id"):
        console.print(f"\n[bold green]REGISTRATION SUCCESSFUL![/bold green]")
        console.print(f"User ID: [bold cyan]{inserted['id']}[/bold cyan]")
        console.print(f"Face embeddings: [bold magenta]3 x 512[/bold magenta]")
        console.print(f"Voice embedding: [bold magenta]1 x 192[/bold magenta]")
        console.print(f"Voice backup saved: [bold yellow]{voice_path}[/bold yellow]")
    else:
        console.print("[bold red]Failed to save user[/bold red]")
        raise typer.Exit(code=1)