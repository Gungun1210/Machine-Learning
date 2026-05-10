# 🎬 AutoStream AI Agent — Inflx Social-to-Lead Platform

> **ML Intern Assignment** | ServiceHive × Inflx | Built with LangGraph + Groq + Flask

A fully agentic conversational AI that converts social media conversations into qualified business leads for **AutoStream** — a fictional SaaS video editing platform for content creators.

---

## 📁 Project Structure

```
inflx-autostream/
│
├── agent/                         # Core agent package
│   ├── __init__.py
│   ├── state.py                   # LangGraph AgentState TypedDict
│   ├── rag.py                     # RAG pipeline — loads & formats knowledge base
│   ├── nodes.py                   # All LangGraph node functions (4 nodes)
│   ├── graph.py                   # Graph builder + ConversationSession manager
│   └── tools.py                   # mock_lead_capture() tool
│
├── knowledge_base/
│   └── autostream_kb.json         # Local RAG knowledge base (plans, policies, FAQs)
│
├── app.py                         # Flask REST API server
├── index.html                     # Frontend UI (served by Flask)
├──cli_test.py                     #Scripted + interactive terminal test           
│
├── requirements.txt
├── .env
└── README.md
```

---

## 🚀 How to Run Locally


### Step 1 — Clone the project

```bash
git clone (https://github.com/Gungun1210/Machine-Learning)
cd inflx-autostream
```
### Step 2 — Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```
### Step 3 - Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure environment

```bash
cp .env.example .env
#Open .env and set values:

GROQ_API_KEY=

GROQ_MODEL=mock
```

### Step 6 — Run the Flask server

```bash
python app.py
```

Open your browser at **http://localhost:5000** 

---

## 🧠 Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it provides **explicit, inspectable state machines** — the graph topology (nodes + edges + routing) is declared upfront, making the agent's decision flow predictable and debuggable. Each conversation turn flows through a directed acyclic graph: `intent_node → extract_node → capture_node → response_node`, with conditional routing deciding which path to take. This deterministic structure is critical for a lead-capture workflow where the tool must **never fire prematurely**.

### How State is Managed

State is a single `AgentState` TypedDict that travels through every node. The `messages` field uses LangGraph's `add_messages` reducer, which **appends** new messages rather than replacing the list — giving the agent full multi-turn memory across 5–6+ conversation turns without any external memory store.

Key state fields:
- `intent` — detected intent, updated every turn
- `collecting_lead` — activated on HIGH_INTENT, never reset until captured
- `lead_name / lead_email / lead_platform` — incrementally populated
- `lead_captured` — boolean guard preventing duplicate tool calls

The RAG pipeline injects the full `autostream_kb.json` as a formatted context block into the system prompt, ensuring every LLM response is grounded in accurate product data.

---

## 📱 WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp using the **WhatsApp Business Cloud API**:

### Architecture

```
User (WhatsApp)
      │
      ▼
WhatsApp Cloud API
      │  POST webhook event (message)
      ▼
ngrok / Cloud Server  →  Flask /webhook endpoint
      │                        │
      │                   ConversationSession.chat()
      │                        │
      │                   Agent graph executes
      │                        │
      │              Response text generated
      │                        │
      ▼                        ▼
WhatsApp Cloud API  ←  POST reply to messages API
      │
      ▼
User receives reply (WhatsApp)
```

### Implementation Steps

**1. Register a WhatsApp Business App**
- Go to [Meta for Developers](https://developers.facebook.com)
- Create an App → Add WhatsApp product
- Get your `WHATSAPP_TOKEN` and `PHONE_NUMBER_ID`

**2. Add a webhook endpoint to `app.py`**

```python
@app.route("/webhook", methods=["GET", "POST"])
def whatsapp_webhook():
    # GET: verify webhook with Meta's challenge
    if request.method == "GET":
        if request.args.get("hub.verify_token") == os.environ["WA_VERIFY_TOKEN"]:
            return request.args.get("hub.challenge")
        return "Forbidden", 403

    # POST: handle incoming message
    data = request.json
    try:
        msg_obj  = data["entry"][0]["changes"][0]["value"]["messages"][0]
        sender   = msg_obj["from"]          # WhatsApp phone number (= session ID)
        text     = msg_obj["text"]["body"]
    except (KeyError, IndexError):
        return "ok", 200                    # Ignore non-message events

    session  = get_session(sender)
    result   = session.chat(text)

    send_whatsapp_message(sender, result["response"])
    return "ok", 200


def send_whatsapp_message(to: str, body: str):
    import requests
    url = f"https://graph.facebook.com/v18.0/{os.environ['WA_PHONE_ID']}/messages"
    headers = {
        "Authorization": f"Bearer {os.environ['WA_TOKEN']}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body}
    }
    requests.post(url, json=payload, headers=headers)
```

**3. Expose locally with ngrok (for testing)**

```bash
ngrok http 5000
# Copy the HTTPS URL → set as webhook in Meta Developer Console
```

**4. Production deployment**

Deploy the Flask app to any cloud (Railway, Render, AWS EC2) with a public HTTPS URL.
For Groq mode, set the GROQ_API_KEY and GROQ_MODEL environment variables so the backend can connect to Groq’s hosted LLMs.
For Mock mode, leave `GROQ_API_KEY` empty and set `GROQ_MODEL=mock` — the app will serve canned responses from the local knowledge base (autostream_kb.json), making it free to demo without GPU or paid API.

**Session isolation** is handled automatically — the sender's WhatsApp phone number is used as the `session_id`, so every user has their own independent conversation state.

---

## ✅ Evaluation Criteria Checklist

| Criterion | Implementation |
|---|---|
| **Intent Detection** | `nodes.py → intent_node()` — 5 intent classes via LLM classifier |
| **RAG Pipeline** | `agent/rag.py` — local JSON KB formatted + injected into system prompt |
| **State Management** | `agent/state.py` — LangGraph `AgentState` TypedDict + `add_messages` reducer |
| **Tool Calling Logic** | `agent/tools.py + nodes.py capture_node()` — guarded, fires only when all 3 fields collected |
| **Code Clarity** | Each concern in its own file; every function has a docstring |
| **Real-world Deployability** | WhatsApp webhook section above; session isolation; `.env` config |

---

## 🔌 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Frontend UI |
| `/api/chat` | POST | `{ session_id, message }` → agent reply |
| `/api/reset` | POST | `{ session_id }` → reset session |
| `/api/health` | GET | Ollama status + active sessions |
| `/api/session-info` | GET | `?session_id=` → current lead state |

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (llama3-70b-8192 or other supported models) — hosted API; mock fallback for free demo|
| Agent Framework | LangGraph (StateGraph) |
| RAG | Local JSON knowledge base + prompt injection |
| Backend | Flask + Flask-CORS |
| Frontend | Vanilla HTML/CSS/JS (no framework needed) |
| Language | Python 3.9+ |
