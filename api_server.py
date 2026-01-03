"""api_server.py

Cloud backend for Face+Voice verification.

IMPORTANT:
- This server must NEVER access webcam/microphone hardware.
- The browser captures image/audio and uploads them.

Render start command (recommended):
  gunicorn api_server:app
"""

from __future__ import annotations

import os
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from config.settings import settings
from db.client import get_db
from db.session_repo import mark_session
from services.embedding import get_face_embedding
from services.comparison import verify_match
from services.face_detection import get_cropped_face
from services.io_helpers import decode_image_bytes
from services.voice_embedding import verify_voice_from_audio_bytes_detailed



from werkzeug.exceptions import HTTPException
import traceback

@app.errorhandler(Exception)
def handle_exception(e):
    # Let HTTP errors (404, 400, etc.) pass through normally
    if isinstance(e, HTTPException):
        return e

    # Log real server-side bugs
    print(traceback.format_exc())

    return {"error": "Internal Server Error"}, 500


@app.route("/favicon.ico")
def favicon():
    return "", 204





def _json_safe(obj: Any):
    """Convert numpy/scalar objects into JSON-safe types."""
    try:
        import numpy as np

        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    # Ensure API endpoints return JSON even on errors
    @app.errorhandler(Exception)
    def _handle_exception(e):
        path = getattr(request, "path", "")
        if path.startswith("/api/"):
            return jsonify({"ok": False, "error": str(e), "type": e.__class__.__name__}), 500
        raise e

    @app.get("/api/health")
    def health():
        db = get_db()
        return jsonify({"ok": True, "db": db is not None})

    @app.post("/api/session/auto")
    def session_auto():
        """Manual flow endpoint.

        Expects multipart/form-data:
          - face: image/jpeg or image/png
          - voice: audio/wav (16-bit PCM recommended)
        """
        face_file = request.files.get("face")
        voice_file = request.files.get("voice")
        if not face_file or not voice_file:
            return jsonify({"ok": False, "error": "face and voice are required"}), 400

        img = decode_image_bytes(face_file.read())
        if img is None:
            return jsonify({"ok": False, "error": "invalid image"}), 400

        cropped = get_cropped_face(img)
        if cropped is None:
            return jsonify({"ok": True, "verified": False, "reason": "no_face_detected"}), 200

        face_emb = get_face_embedding(cropped)
        if face_emb is None:
            return jsonify({"ok": True, "verified": False, "reason": "face_embedding_failed"}), 200

        face_result = verify_match(face_emb)
        stored_voice = face_result.get("voice_embedding")

        # Privacy: never send stored embedding to browser
        if isinstance(face_result, dict):
            face_result.pop("voice_embedding", None)

        if not face_result.get("matched"):
            return jsonify({"ok": True, "verified": False, "reason": "face_not_matched", "face": _json_safe(face_result)}), 200

        if not stored_voice:
            return jsonify({"ok": True, "verified": False, "reason": "no_voice_enrolled", "face": _json_safe(face_result)}), 200

        score, passed, voice_reason = verify_voice_from_audio_bytes_detailed(
            stored_emb=stored_voice,
            audio_bytes=voice_file.read(),
            threshold=settings.voice_threshold,
        )

        response = {
            "ok": True,
            "verified": bool(passed),
            "reason": "ok" if passed else voice_reason,
            "face": _json_safe(face_result),
            "voice": {"score": float(score), "passed": bool(passed)},
        }

        if passed:
            user_id = face_result.get("user_id")
            name = face_result.get("name")
            email = face_result.get("email")
            action, message = mark_session(user_id, name, email)
            response["session"] = {"action": action, "message": message}

        return jsonify(response)

    # Minimal UI
    WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "verify.html")

    @app.get("/web/<path:filename>")
    def web_static(filename: str):
        return send_from_directory(WEB_DIR, filename)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
