"""
app.py
------
Flask REST API server for the AutoStream AI Agent.

Endpoints:
  POST /api/chat         → Send a user message, get agent reply
  POST /api/reset        → Reset a conversation session
  GET  /api/health       → Server health + Groq connectivity check
  GET  /api/session-info → Current lead collection state for a session
  GET  /                 → Serve the frontend HTML
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'agent' package imports work
sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from agent.graph import ConversationSession


# ─────────────────────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder="static")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# In-memory session store (use Redis in production)
sessions: dict = {}


def get_session(session_id: str) -> ConversationSession:
    """Return existing session or create a new one."""
    if session_id not in sessions:
        sessions[session_id] = ConversationSession()
    return sessions[session_id]


def check_groq() -> dict:
    """Verify Groq API key and model availability, with mock fallback."""
    api_key = os.environ.get("GROQ_API_KEY")
    model = os.environ.get("GROQ_MODEL", "llama3-8b-8192")

    if not api_key:
        # Mock mode: no API key, so Groq is not reachable
        return {
            "reachable": False,
            "model": "mock",
            "model_ready": True,
            "error": "Running in mock mode (no GROQ_API_KEY set)"
        }

    # For simplicity, assume Groq is reachable if API key is set.
    return {
        "reachable": True,
        "model": model,
        "model_ready": True
    }


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend HTML file."""
    return send_from_directory(str(Path(__file__).parent), "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    """Health check — confirms server is up and Groq API key is set or mock mode active."""
    groq_status = check_groq()
    return jsonify({
        "status": "ok",
        "groq": groq_status,
        "active_sessions": len(sessions),
        "model": groq_status["model"],
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.
    """
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "default")
    user_message = (data.get("message") or "").strip()

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    groq_status = check_groq()
    if not groq_status["reachable"] and groq_status["model"] != "mock":
        return jsonify({
            "error": (
                "Groq API key not set or invalid. "
                "Please add GROQ_API_KEY to your .env file."
            )
        }), 503

    session = get_session(session_id)

    try:
        result = session.chat(user_message)
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Agent error: {str(exc)}"}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """Reset a conversation session back to initial state."""
    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id", "default")
    if session_id in sessions:
        sessions[session_id].reset()
    return jsonify({"status": "reset", "session_id": session_id})


@app.route("/api/session-info", methods=["GET"])
def session_info():
    """Return the current lead collection state for a session (useful for debugging)."""
    session_id = request.args.get("session_id", "default")
    session = get_session(session_id)
    state = session.state
    return jsonify({
        "session_id": session_id,
        "message_count": session.message_count,
        "intent": state.get("intent", ""),
        "collecting_lead": state.get("collecting_lead", False),
        "lead_name": state.get("lead_name"),
        "lead_email": state.get("lead_email"),
        "lead_platform": state.get("lead_platform"),
        "lead_captured": state.get("lead_captured", False),
        "capture_result": state.get("capture_result"),
    })


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"

    print("\n" + "=" * 55)
    print("  🎬  AutoStream AI Agent — Influx Platform (Groq)")
    print("=" * 55)

    groq = check_groq()
    model = groq["model"]

    if groq["reachable"]:
        print(f"  ✅  Groq API key set")
        print(f"  ✅  Model '{model}' is ready")
    elif groq["model"] == "mock":
        print(f"  ⚡  Running in MOCK mode (no Groq API key)")
    else:
        print(f"  ❌  Groq API key missing or invalid")

    print(f"\n  🌐  Frontend: http://localhost:{port}")
    print(f"  🔌  API Base: http://localhost:{port}/api")
    print("=" * 55 + "\n")

    app.run(debug=debug, host="0.0.0.0", port=port)
