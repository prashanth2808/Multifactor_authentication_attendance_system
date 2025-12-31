# main.py
"""
Face + Voice Login/Logout System - CLI
Run: python main.py <command>
"""

import typer
from cli.register import register as register_cmd
from cli.session import session as session_cmd      # ← Login / Logout
from cli.report import report as report_cmd          # ← Daily report
from cli.admin import app as admin_app

app = typer.Typer(
    name="Biometric Login/Logout CLI",
    help="Secure face + voice login/logout system",
    no_args_is_help=True,
)

# CORE COMMANDS
app.command()(register_cmd)           # python main.py register
app.command("session")(session_cmd)   # ← python main.py session (Login/Logout)
app.command("report")(report_cmd)     # ← python main.py report

# ADMIN SUB-APP
app.add_typer(admin_app, name="admin", help="View users, sessions, export")

@app.command()
def version():
    """Show version"""
    typer.echo("Biometric Login/Logout CLI v2.0.0")

if __name__ == "__main__":
    app()