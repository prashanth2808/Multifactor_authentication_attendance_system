# cli/attendance.py
"""
Mark attendance using webcam + microphone (Face + Voice REQUIRED)
Command: python main.py attendance
"""

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from utils.camera import capture_face_image
from services.embedding import get_face_embedding
from services.comparison import verify_match
from services.voice_embedding import verify_voice_live  # ← NEW
# NOTE: Attendance logging repo not implemented in this version.
# Use `python main.py session` for login/logout based attendance.
from config.settings import settings
import numpy as np

console = Console()

def scan():
    """
    Capture face + voice, require BOTH to match, mark attendance
    """
    console.print("[bold blue]Starting attendance scan...[/bold blue]")
    console.print("[cyan]Look at the camera and press SPACE when ready[/cyan]")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        progress.add_task("Capturing face...", total=None)
        img = capture_face_image()
        if img is None:
            console.print("[bold red]Capture failed or cancelled[/bold red]")
            raise typer.Exit(code=1)

    console.print("[green]Face captured! Processing...[/green]")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        progress.add_task("Generating face embedding...", total=None)
        embedding = get_face_embedding(img)
        if embedding is None:
            console.print("[bold red]Failed to generate face embedding[/bold red]")
            raise typer.Exit(code=1)

    console.print("[green]Face embedding generated (512-dim)[/green]")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), transient=True) as progress:
        progress.add_task("Comparing faces with database...", total=None)
        face_result = verify_match(embedding)

    if not face_result["matched"]:
        _show_no_face_match(face_result.get("confidence", 0.0))
        return

    # === FACE MATCHED → NOW VERIFY VOICE ===
    candidate_user = face_result
    face_conf = face_result["confidence"]

    console.print(f"\n[bold yellow]Face matched: {candidate_user['name']} ({face_conf:.1%})[/bold yellow]")
    console.print("[cyan]Now verifying voice... Please say your name clearly[/cyan]")

    # Load stored voice embedding from user
    stored_voice_emb = candidate_user.get("voice_embedding")
    if not stored_voice_emb:
        console.print("[bold red]Voice not enrolled for this user[/bold red]")
        _show_access_denied("Voice not registered")
        return

    stored_voice_emb = np.array(stored_voice_emb)

    try:
        voice_score, voice_passed = verify_voice_live(stored_voice_emb, duration=5.0, threshold=0.68)
    except Exception as e:
        console.print(f"[red]Voice verification failed: {e}[/red]")
        _show_access_denied("Voice error")
        return

    # === FINAL DECISION: BOTH REQUIRED ===
    if face_conf >= 0.65 and voice_passed:  # Face threshold slightly lower than before
        _show_success(candidate_user, face_conf, voice_score)
        # Attendance logging not wired here; session-based attendance is handled by `mark_session()`.
        console.print("[dim]Attendance logged via session workflow only in this version.[/dim]")
    else:
        _show_access_denied(f"Face: {face_conf:.1%}, Voice: {voice_score:.1%}")

    console.print("\n[bold]Scan complete![/bold]")


# === HELPER: NO FACE MATCH ===
def _show_no_face_match(best_conf):
    console.print(f"\n[bold red]ACCESS DENIED[/bold red]")
    console.print(f"[bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")
    console.print(f"Status: [bold red]Unknown Face[/bold red]")
    if best_conf > 0:
        console.print(f"Best Match: [red]{best_conf:.1%}[/red] (Below threshold)")
    console.print(f"Required: [yellow]{settings.similarity_threshold:.1%}[/yellow]")
    console.print(f"Suggestion: [dim]Register or improve lighting[/dim]")
    _print_time()
    console.print(f"[bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")


# === HELPER: SUCCESS ===
def _show_success(user, face_conf, voice_conf):
    name = user["name"]
    email = user["email"]

    console.print(f"\n[bold green]ATTENDANCE RECORDED![/bold green]")
    console.print(f"[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    console.print(f"Name: [bold yellow]{name}[/bold yellow]")
    console.print(f"Email: [cyan]{email}[/cyan]")

    # Face confidence
    face_color = "bold green" if face_conf >= 0.85 else "green" if face_conf >= 0.75 else "yellow"
    console.print(f"Face: [{face_color}]{face_conf:.1%}[/{face_color}]")

    # Voice confidence
    voice_color = "bold green" if voice_conf >= 0.85 else "green" if voice_conf >= 0.75 else "yellow"
    console.print(f"Voice: [{voice_color}]{voice_conf:.1%}[/{voice_color}]")

    console.print(f"Method: [bold magenta]Face + Voice[/bold magenta]")
    _print_time()
    console.print(f"[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")


# === HELPER: ACCESS DENIED ===
def _show_access_denied(reason=""):
    console.print(f"\n[bold red]ACCESS DENIED[/bold red]")
    console.print(f"[bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")
    console.print(f"Status: [bold red]Verification Failed[/bold red]")
    if reason:
        console.print(f"Reason: [red]{reason}[/red]")
    _print_time()
    console.print(f"[bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")


# === HELPER: TIME ===
def _print_time():
    from datetime import datetime
    now = datetime.now().strftime("%I:%M:%S %p")
    console.print(f"Time: [dim]{now}[/dim]")