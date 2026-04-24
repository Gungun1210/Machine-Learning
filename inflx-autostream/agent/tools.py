"""
tools.py
--------
Tool definitions for the AutoStream AI Agent.
mock_lead_capture is the primary tool triggered when
all three lead fields (name, email, platform) are collected.
"""

import json
import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# MOCK LEAD CAPTURE — Required by assignment spec
# ─────────────────────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock CRM API that simulates lead capture to a backend database.

    In production this would POST to a real CRM like HubSpot or Salesforce.
    For this assignment it prints a confirmation and returns a structured payload.

    Args:
        name     (str): Full name of the prospect
        email    (str): Email address of the prospect
        platform (str): Content platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        dict: Success payload with lead_id and timestamp
    """
    timestamp = datetime.datetime.now().isoformat()
    lead_id = f"LEAD-{abs(hash(email + name)) % 100000:05d}"

    # --- Required print statement from assignment ---
    print(f"\n{'='*55}")
    print(f"  [TOOL CALLED] mock_lead_capture()")
    print(f"  Lead captured successfully: {name}, {email}, {platform}")
    print(f"  Lead ID  : {lead_id}")
    print(f"  Timestamp: {timestamp}")
    print(f"{'='*55}\n")

    # Optionally persist to a local leads.json for demo purposes
    _persist_lead(lead_id, name, email, platform, timestamp)

    return {
        "status": "success",
        "lead_id": lead_id,
        "name": name,
        "email": email,
        "platform": platform,
        "timestamp": timestamp,
        "message": f"Lead captured successfully: {name}, {email}, {platform}"
    }


def _persist_lead(lead_id, name, email, platform, timestamp):
    """Append the captured lead to leads.json for traceability."""
    leads_path = Path(__file__).parent.parent / "leads.json"
    leads = []
    if leads_path.exists():
        try:
            with open(leads_path, "r") as f:
                leads = json.load(f)
        except Exception:
            leads = []

    leads.append({
        "lead_id": lead_id,
        "name": name,
        "email": email,
        "platform": platform,
        "timestamp": timestamp,
    })

    with open(leads_path, "w") as f:
        json.dump(leads, f, indent=2)