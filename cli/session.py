# cli/session.py
"""
FULLY AUTOMATIC BIOMETRIC SESSION — FINAL VERSION
Strict 9-hour fault rule + Malpractice protection
"""

import typer
import cv2
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box
from datetime import datetime

from services.embedding import get_face_embedding
from services.comparison import verify_match
from services.voice_embedding import verify_voice_live
from services.face_detection import get_cropped_face
from db.session_repo import mark_session
from config.settings import settings

console = Console()

def session():
    console.print("\n[bold blue]AUTOMATIC BIOMETRIC SESSION STARTED[/bold blue]")
    console.print("[cyan]Sit in front of camera — system recognizes you instantly[/cyan]\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        console.print("[bold red]ERROR: Cannot open webcam[/bold red]")
        return

    status_text = Text("Scanning for face...", style="bold yellow")
    with Live(Panel(status_text, box=box.ROUNDED, border_style="bright_blue"), refresh_per_second=4) as live:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cropped = get_cropped_face(frame)
            if cropped is None:
                live.update(Panel("No face detected", border_style="red"))
                cv2.imshow('Biometric Session - ESC to exit', frame)
                if cv2.waitKey(1) == 27:
                    break
                continue

            embedding = get_face_embedding(cropped)
            if embedding is None:
                continue

            result = verify_match(embedding, threshold=0.60)
            if result["matched"]:
                user = result
                name = user["name"]
                email = user["email"]
                conf = user["confidence"]

                status_text = Text("FACE RECOGNIZED!\n", style="bold green")
                status_text.append(f"{name}\n", style="bold yellow")
                status_text.append(f"{email}\n", style="cyan")
                status_text.append(f"Confidence: {conf:.1%}", style="bright_white")
                live.update(Panel(status_text, border_style="green", box=box.DOUBLE))

                cap.release()
                cv2.destroyAllWindows()
                break
            else:
                best = result.get("confidence", 0)
                msg = f"Unknown person ({best:.1%})" if best > 0.3 else "Unknown person"
                live.update(Panel(msg, border_style="red"))

            cv2.imshow('Biometric Session - ESC to exit', frame)
            if cv2.waitKey(1) == 27:
                cap.release()
                cv2.destroyAllWindows()
                console.print("[dim]Session cancelled[/dim]")
                return

    # === FACE VERIFIED — VOICE CONFIRMATION ===
    console.print(f"\n[bold green]Hello {name} — Face verified ({conf:.1%})[/bold green]")
    console.print("[bold cyan]Press V to continue with voice verification (or N to cancel)[/bold cyan]")

    while True:
        choice = console.input("[bold]→ V/N: [/bold]").strip().lower()
        if choice in ["v", ""]:
            break
        if choice == "n":
            console.print("[yellow]Session cancelled by user[/yellow]")
            return
        console.print("[red]Invalid — Press V or N[/red]")

    # === VOICE VERIFICATION ===
    stored_voice = np.array(user["voice_embedding"])
    console.print("\n[bold blue]Speak your phrase now...[/bold blue]")
    voice_score, voice_passed = verify_voice_live(stored_voice, duration=5.0, threshold=0.68)

    if not voice_passed:
        console.print(f"\n[bold red]VOICE VERIFICATION FAILED ({voice_score:.1%})[/bold red]")
        console.print("[bold red]ACCESS DENIED — Voice not matched[/bold red]")
        _print_time()
        return

    console.print(f"[bold green]VOICE VERIFIED! ({voice_score:.1%})[/bold green]")

    # === FINAL ACTION: LOGIN / LOGOUT / MALPRACTICE / AUTO-ABSENT ===
    # Debug: Show what keys are available in user object
    console.print(f"[dim]Debug: User object keys: {list(user.keys())}[/dim]")
    
    user_id = user.get("user_id")
    if user_id is None:
        console.print(f"[red]Error: User ID not found in verification result[/red]")
        console.print(f"[yellow]Available keys: {list(user.keys())}[/yellow]")
        return
    
    action, message = mark_session(user_id, name, email)

    if action == "LOGIN":
        _show_login_success(name, email, conf, voice_score, message)

    elif action == "LOGOUT":
        _show_logout_success(name, conf, voice_score, message)

    elif action == "MALPRACTICE":
        _show_malpractice_warning(name, message)

    elif action == "ABSENT_AUTO":
        _show_auto_absent(name, message)

    else:
        console.print(f"[yellow]{message}[/yellow]")

    _print_time()
    console.print("\n[bold]Session completed.[/bold]\n")


# === BEAUTIFUL SUCCESS MESSAGES ===
def _show_login_success(name, email, face_conf, voice_conf, msg):
    console.print(f"\n[bold green]LOGIN SUCCESSFUL![/bold green]")
    console.print(f"[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    console.print(f"Welcome: [bold yellow]{name}[/bold yellow] ({email})")
    console.print(f"Face: [green]{face_conf:.1%}[/green] | Voice: [green]{voice_conf:.1%}[/green]")
    console.print(f"Status: [bold green]Logged In — Present Today[/bold green]")
    console.print(f"[dim]{msg}[/dim]")
    console.print(f"[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

def _show_logout_success(name, face_conf, voice_conf, msg):
    console.print(f"\n[bold magenta]LOGOUT SUCCESSFUL![/bold magenta]")
    console.print(f"[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")
    console.print(f"Goodbye: [bold yellow]{name}[/bold yellow]")
    console.print(f"Face: [green]{face_conf:.1%}[/green] | Voice: [green]{voice_conf:.1%}[/green]")
    console.print(f"Status: [bold green]Present Today[/bold green]")
    console.print(f"[magenta]{msg}[/magenta]")
    console.print(f"[bold cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold cyan]")

def _show_malpractice_warning(name, msg):
    console.print(f"\n[bold red]MALPRACTICE DETECTED[/bold red]")
    console.print(f"[bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")
    console.print(f"User: [bold yellow]{name}[/bold yellow]")
    console.print(f"[bold red]{msg}[/bold red]")
    console.print(f"[dim]No action taken — already marked today[/dim]")
    console.print(f"[bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")

def _show_auto_absent(name, msg):
    console.print(f"\n[bold red]MARKED ABSENT AUTOMATICALLY[/bold red]")
    console.print(f"[bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")
    console.print(f"User: [bold yellow]{name}[/bold yellow]")
    console.print(f"[bold red]{msg}[/bold red]")
    console.print(f"[dim]You forgot to logout for 9+ hours[/dim]")
    console.print(f"[bold red]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[/bold red]")

def _print_time():
    now = datetime.now().strftime("%I:%M:%S %p • %d %b %Y")
    console.print(f"[dim]{now}[/dim]")