# cli/report.py
"""
View login/logout report
Command: python main.py report [today | --date YYYY-MM-DD]
"""

import typer
from rich.console import Console
from rich.table import Table
from rich import box
from datetime import datetime
from db.session_repo import get_report
from typing import Optional

console = Console()

def report(
    today: bool = typer.Option(False, "--today", "-t", help="Show today's report"),
    date: Optional[str] = typer.Option(None, "--date", "-d", help="Report for specific date (YYYY-MM-DD)")
):
    """
    Display login/logout report in a clean table
    """
    if today and date:
        console.print("[red]Error: Use either --today or --date, not both[/red]")
        raise typer.Exit(code=1)

    if today:
        report_date = datetime.now().strftime("%Y-%m-%d")
        title = f"LOGIN/LOGOUT REPORT - TODAY ({report_date})"
    elif date:
        try:
            datetime.strptime(date, "%Y-%m-%d")
            report_date = date
            title = f"LOGIN/LOGOUT REPORT - {date}"
        except ValueError:
            console.print("[red]Invalid date format. Use YYYY-MM-DD[/red]")
            raise typer.Exit(code=1)
    else:
        # Default: show today
        report_date = datetime.now().strftime("%Y-%m-%d")
        title = f"LOGIN/LOGOUT REPORT - TODAY ({report_date})"

    # Fetch data
    sessions = get_report(report_date)

    if not sessions:
        console.print(f"[yellow]No login/logout records found for {report_date}[/yellow]")
        return

    # Create table
    table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Name", style="cyan", justify="left")
    table.add_column("Email", style="dim")
    table.add_column("Login", style="green")
    table.add_column("Logout", style="red")
    table.add_column("Duration", style="yellow", justify="center")
    table.add_column("Status", style="bold")

    for s in sessions:
        # db.session_repo.get_report returns formatted strings
        table.add_row(
            s.get("name", ""),
            s.get("email", ""),
            s.get("login", "—"),
            s.get("logout", "—"),
            s.get("duration", "—"),
            s.get("status", "—"),
        )

    console.print(table)
    console.print(f"\n[dim]Total users: {len(sessions)}[/dim]")