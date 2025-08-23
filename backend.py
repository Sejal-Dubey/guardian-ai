#!/usr/bin/env python3
"""
Real-time Email Security Backend – Groq Production Edition
"""
import asyncio
import base64
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List

import google.oauth2.credentials
import google.auth.transport.requests as g_requests
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import PlainTextResponse
from google.auth.transport import requests
from google.oauth2 import id_token
from googleapiclient.discovery import build
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

# Add shared to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import get_settings
from shared.crypto import decrypt_token
from shared.llm import chat_completion

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] backend: %(message)s",
    handlers=[
        RotatingFileHandler("backend.log", maxBytes=5_000_000, backupCount=3, encoding='utf-8'),
        logging.StreamHandler(),
    ],
)

USER_DB_FILE = "user_database.json"

def load_db() -> Dict:
    if os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE) as f:
            return json.load(f)
    return {}

def save_db(db: Dict):
    with open(USER_DB_FILE, "w") as f:
        json.dump(db, f, indent=2)

USER_DB = load_db()

# ---------------------------------------------------------------------------
class Persona(BaseModel):
    inferred_role: str
    primary_topics: List[str]
    key_interlocutors: List[str]
    risk_indicators_to_watch_for: List[str]

class RegRequest(BaseModel):
    email: str
    refresh_token: str
    persona: Persona

# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=30))
def get_gmail_service(email: str):
    data = USER_DB[email]
    info = {
        "refresh_token": decrypt_token(data["refresh_token"]),
        "client_id": settings.google_client_id,
        "client_secret": settings.google_client_secret,
        "token_uri": "https://oauth2.googleapis.com/token",
    }
    creds = google.oauth2.credentials.Credentials.from_authorized_user_info(info)
    return build("gmail", "v1", credentials=creds)

# ---------------------------------------------------------------------------
class ConnMgr:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

mgr = ConnMgr()
app = FastAPI(title="Email Security Backend – Groq")

# ---------------------------------------------------------------------------
@app.post("/register")
async def register(req: RegRequest, bg: BackgroundTasks):
    if req.email in USER_DB:
        raise HTTPException(status_code=409, detail="Already registered")
    USER_DB[req.email] = {
        "refresh_token": req.refresh_token,
        "persona": req.persona.dict(),
        "history_id": None,
        "last_refresh": datetime.utcnow().isoformat(),
    }
    save_db(USER_DB)
    bg.add_task(setup_watch, req.email)
    return {"message": "registered"}

@app.get("/status/{email}")
async def status(email: str):
    if email in USER_DB:
        return {"status": "registered"}
    raise HTTPException(status_code=404, detail="not found")

# ---------------------------------------------------------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=30))
async def setup_watch(email: str):
    svc = get_gmail_service(email)
    body = {
        "labelIds": ["INBOX"],
        "topicName": f"projects/{settings.google_project_id}/topics/{settings.pubsub_topic_name}",
    }
    resp = svc.users().watch(userId=email, body=body).execute()
    USER_DB[email]["history_id"] = resp["historyId"]
    save_db(USER_DB)
    logging.info("Watch set for %s", email)

# ---------------------------------------------------------------------------
@app.post("/gmail/webhook")
async def gmail_webhook(req: Request):
    auth_header = req.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        logging.error("Missing or invalid Authorization header")
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = auth_header.split("Bearer ")[1]
    try:
        claim = id_token.verify_oauth2_token(token, g_requests.Request())
        if claim['iss'] not in ['https://accounts.google.com', 'accounts.google.com']:
             raise ValueError('Wrong issuer.')
    except Exception as e:
        logging.error(f"JWT verification failed: {e}")
        raise HTTPException(status_code=401, detail=f"Bad JWT: {e}")

    envelope = await req.json()
    if "message" not in envelope or "data" not in envelope["message"]:
        logging.error("Invalid Pub/Sub message format")
        return PlainTextResponse("Bad Request: Invalid Pub/Sub message", status_code=400)

    data = json.loads(base64.b64decode(envelope["message"]["data"]))
    email = data["emailAddress"]
    new_hist = data["historyId"]

    if email not in USER_DB:
        logging.warning(f"Webhook received for unknown user: {email}")
        return PlainTextResponse("OK")

    svc = get_gmail_service(email)
    hist = (
        svc.users()
        .history()
        .list(userId=email, startHistoryId=USER_DB[email]["history_id"])
        .execute()
    )
    for h in hist.get("history", []):
        for m_added in h.get("messagesAdded", []):
            msg_id = m_added["message"]["id"]
            msg = svc.users().messages().get(userId=email, id=msg_id, format="full").execute()
            headers = {h["name"].lower(): h["value"] for h in msg["payload"]["headers"]}
            email_data = {
                "from": headers.get("from", ""),
                "subject": headers.get("subject", ""),
                "snippet": msg.get("snippet", ""),
                "auth": headers.get("authentication-results", ""),
            }
            asyncio.create_task(analyze(email, email_data))

    USER_DB[email]["history_id"] = new_hist
    save_db(USER_DB)
    return PlainTextResponse("OK")

# ---------------------------------------------------------------------------
class DebugAnalyzeRequest(BaseModel):
    email: str
    email_data: Dict[str, str]

@app.post("/debug/analyze-payload")
async def debug_analyze_payload(req: DebugAnalyzeRequest):
    if req.email not in USER_DB:
        raise HTTPException(status_code=404, detail="User not found in DB")
    logging.info(f"--- Received debug analysis request for user: {req.email} ---")
    asyncio.create_task(analyze(req.email, req.email_data))
    return {"message": "Analysis task triggered successfully"}

# ---------------------------------------------------------------------------
async def analyze(user_email: str, email_data: Dict):
    persona = USER_DB[user_email]["persona"]
    prompt = f"""
Your instructions are to act as a **Tier-1 SOC analyst with a very low tolerance for risk**. Your task is to analyze the email provided in the <email_data> tags and return a valid JSON object. Assume any suspicious indicator could be part of a sophisticated attack.

First, reason through the potential threats in a step-by-step manner based on the <detection_criteria> and the user's <persona_context>.

Second, based on your reasoning, determine the single most likely threat type.

Third, conclude if the email is a threat.

Fourth, create a user_summary: a single, non-technical sentence explaining the danger to the user. For safe emails, this summary should be reassuring.

**IMPORTANT RULE:** If the `threat_type` is "Safe", then the `action` MUST also be "Safe".
<persona_context>
{json.dumps(persona, indent=2)}
</persona_context>

<email_data>
From: {email_data['from']}
Subject: {email_data['subject']}
Auth: {email_data['auth']}
Body snippet: {email_data['snippet'][:400]}...
</email_data>

<detection_criteria>
- SPF/DKIM/DMARC failures, look-alike domains, homoglyphs.
- Social engineering tactics: Urgency, threats, fake invoices, credential harvesting lures.
- Mismatch between sender and links.
</detection_criteria>

Return **only** a valid JSON object with the following structure:
{{
  "reasoning": [
    "A step-by-step analysis point.",
    "Another analysis point.",
    "And so on..."
  ],
  "threat_type": "Spear Phishing|Malware|Credential Harvesting|Social Engineering|Safe",
  "is_threat": <bool>,
  "confidence": 0.0-1.0,
  "action": "Medium Risk|High Risk|Low Risk|Safe",
  "user_summary": "A simple, one-sentence explanation for the end-user."
}}
""".strip()
    
    resp = "" 
    try:
        resp = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=700, 
            temperature=0.1,
        )
        
        json_start = resp.find('{')
        json_end = resp.rfind('}')
        if json_start != -1 and json_end != -1:
            json_str = resp[json_start:json_end+1]
            data = json.loads(json_str)
            
            # --- THIS IS THE KEY CHANGE ---
            # Always create and broadcast the payload, regardless of threat level.
            alert_payload = {
                "user": user_email,
                "email": email_data,
                "alert": data
            }
            await mgr.broadcast(json.dumps(alert_payload))

            # Log threats as warnings and safe emails as info.
            if data.get("is_threat"):
                logging.warning("Threat for %s: %s", user_email, data["reasoning"])
            else:
                logging.info("Safe email processed for %s: %s", user_email, data.get("subject"))
        else:
            logging.error("No JSON object found in LLM response: %s", resp)

    except Exception as e:
        logging.error("Groq error or JSON parsing failed. Raw response: %s", resp)
        logging.exception("Exception details: %s", e)

# ---------------------------------------------------------------------------
@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket, token: str):
    if token != settings.ws_auth_token:
        await websocket.close(code=1008)
        return
    await mgr.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        mgr.disconnect(websocket)

# ---------------------------------------------------------------------------
async def refresh_personas():
    while True:
        await asyncio.sleep(3600)
        now = datetime.utcnow()
        for email, data in list(USER_DB.items()):
            last = datetime.fromisoformat(data["last_refresh"])
            if now - last < timedelta(days=14):
                continue
            try:
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
                from plugin.plugin import PersonaOnboardingPlugin
                PersonaOnboardingPlugin().run()
            except Exception as e:
                logging.exception("Persona refresh failed: %s", e)

# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup():
    for email in USER_DB:
        asyncio.create_task(setup_watch(email))
    asyncio.create_task(refresh_personas())
