# cli/admin.py
"""
ADMIN PANEL — FINAL VERSION
Shows correct Present/Absent with 9-hour fault rule
"""

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from db.user_repo import get_all_users
from db.session_repo import get_report, get_today_status
from datetime import datetime
import csv

app = typer.Typer(help="Admin dashboard — Final Version")

console = Console()

@app.command()
def users():
    """List all registered users"""
    console.print(Panel("[bold blue]REGISTERED USERS[/bold blue]", box=box.DOUBLE))
    users_list = get_all_users()
    if not users_list:
        console.print("[yellow]No users registered[/yellow]")
        return

    table = Table(box=box.ROUNDED)
    table.add_column("Name", style="cyan", justify="left")
    table.add_column("Email", style="green")
    table.add_column("Photos", justify="center")
    table.add_column("Voice", justify="center")
    table.add_column("Registered", style="dim")

    for u in users_list:
        photos = str(u.get("photo_count", 0))
        voice = "[bold green]Yes[/bold green]" if u.get("voice_embedding") else "[bold red]No[/bold red]"
        reg = u["registered_at"].strftime("%d %b %Y") if isinstance(u["registered_at"], datetime) else "—"
        table.add_row(u["name"], u["email"], photos, voice, reg)

    console.print(table)
    console.print(f"\n[bold]Total: {len(users_list)} users[/bold]")

@app.command()
def today():
    """Today’s attendance summary — FINAL 9-HOUR LOGIC"""
    today_str = datetime.now().strftime("%Y-%m-%d")
    console.print(Panel(f"[bold blue]TODAY'S ATTENDANCE — {today_str}[/bold blue]", box=box.DOUBLE))

    all_users = get_all_users()
    if not all_users:
        console.print("[yellow]No users in system[/yellow]")
        return

    sessions = get_report(today_str)

    table = Table(box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Email", style="green")
    table.add_column("Login", justify="center")
    table.add_column("Logout", justify="center")
    table.add_column("Duration", justify="center", style="yellow")
    table.add_column("Status", justify="center", style="bold")

    present = 0
    absent = 0

    for user in all_users:
        user_id = str(user["id"])
        name = user["name"]
        email = user["email"]

        session = next((s for s in sessions if s["email"] == email), None)

        if session:
            login = session["login"]
            logout = session["logout"]
            duration = session["duration"]
            status_text = session["status"]
            if "Present" in status_text:
                present += 1
            elif "Absent" in status_text:
                absent += 1
        else:
            # No session record → check real-time status
            status_info = get_today_status(user_id)
            login = logout = duration = "—"
            if status_info["status"] == "present":
                status_text = "[bold green]Present[/bold green]"
                present += 1
            else:
                status_text = "[bold red]Absent[/bold red]"
                absent += 1
                continue  # skip adding row if no login

        table.add_row(name, email, login, logout, duration, status_text)

    console.print(table)

    # Final summary
    total = len(all_users)
    present = sum(1 for row in table.rows if "Present" in str(row[-1]))
    absent = total - present

    console.print(f"\n[bold cyan]Total Users:[/bold cyan] {total}")
    console.print(f"[bold green]Present:[/bold green] {present}")
    console.print(f"[bold red]Absent:[/bold red] {absent}")
    console.print(f"[dim]Rule: No logout for 9+ hours → Absent (user fault)[/dim]")

@app.command()
def logs(date: str = typer.Option(None, "--date", "-d")):
    """View detailed logs for any date"""
    target = date or datetime.now().strftime("%Y-%m-%d")
    console.print(Panel(f"[bold blue]SESSION LOGS — {target}[/bold blue]", box=box.DOUBLE))

    data = get_report(target)
    if not data:
        console.print("[yellow]No records found[/yellow]")
        return

    table = Table(box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Email", style="green")
    table.add_column("Login", justify="center")
    table.add_column("Logout", justify="center")
    table.add_column("Duration", justify="center", style="yellow")
    table.add_column("Final Status", justify="center", style="bold")

    for row in data:
        table.add_row(
            row["name"],
            row["email"],
            row["login"],
            row["logout"],
            row["duration"],
            row["status"]
        )

    console.print(table)
    console.print(f"\n[dim]Showing {len(data)} records for {target}[/dim]")

@app.command()
def export(date: str = typer.Option(None, "--date", "-d"), file: str = "attendance_report.csv"):
    """Export today's or any date's report"""
    target = date or datetime.now().strftime("%Y-%m-%d")
    data = get_report(target)

    with open(file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Email", "Login", "Logout", "Duration", "Status"])
        for row in data:
            status_clean = row["status"].replace("[bold green]", "").replace("[/bold green]", "") \
                                      .replace("[bold red]", "").replace("[/bold red]", "")
            writer.writerow([row["name"], row["email"], row["login"], row["logout"], row["duration"], status_clean])

    console.print(f"[bold green]Exported to {file}[/bold green]")