#!/usr/bin/env python3
"""
Email Security Persona Onboarding Plugin – Groq Production Edition
"""
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Tuple

import requests
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from tenacity import retry, stop_after_attempt, wait_exponential

# Add shared to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from shared.config import get_settings
from shared.crypto import encrypt_token
from shared.llm import chat_completion

settings = get_settings()
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] plugin: %(message)s",
    handlers=[
        RotatingFileHandler("plugin.log", maxBytes=5_000_000, backupCount=3),
        logging.StreamHandler(),
    ],
)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
CRED_FILE = "credentials.json"
TOKEN_FILE = "token.pickle"

# ---------------------------------------------------------------------------

class PersonaOnboardingPlugin:
    def __init__(self):
        if not os.path.exists(CRED_FILE):
            raise FileNotFoundError(CRED_FILE)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=30))
    def _auth(self) -> Tuple[Any, Any, str]:
        creds = None
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, "rb") as token:
                creds = pickle.load(token)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(CRED_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            with open(TOKEN_FILE, "wb") as token:
                pickle.dump(creds, token)

        service = build("gmail", "v1", credentials=creds)
        profile = service.users().getProfile(userId="me").execute()
        return service, creds, profile["emailAddress"]

    def _already_registered(self, email: str) -> bool:
        try:
            r = requests.get(f"{settings.backend_url}/status/{email}", timeout=10)
            return r.status_code == 200
        except Exception:
            return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=30))
    def _fetch_samples(self, service: Any, n: int = 25) -> List[Dict]:
        resp = (
            service.users()
            .messages()
            .list(
                userId="me",
                maxResults=n,
                q="-category:(promotions,social,updates) in:(inbox,sent)",
            )
            .execute()
        )
        msgs = resp.get("messages", [])
        samples = []
        for m in msgs:
            msg = (
                service.users()
                .messages()
                .get(
                    userId="me",
                    id=m["id"],
                    format="metadata",
                    metadataHeaders=["From", "Subject"],
                )
                .execute()
            )
            headers = {
                h["name"].lower(): h["value"]
                for h in msg["payload"]["headers"]
            }
            samples.append(
                {
                    "from": headers.get("from", ""),
                    "subject": headers.get("subject", ""),
                    "snippet": msg.get("snippet", ""),
                }
            )
        return samples

    def _extract_topics(self, sample: Dict) -> List[str]:
        prompt = f"""
You are an email analyst.
Given the following metadata, return a single line of JSON with a "topics" key. The value should be a list of 3-5 comma-separated, generic professional keywords (NO PII).

From: {sample['from']}
Subject: {sample['subject']}
Snippet: {sample['snippet']}
""".strip()
        # --- THIS IS THE FIX: Increased max_tokens from 60 to 150 ---
        resp = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.1,
        )
        try:
            data = json.loads(resp)
            topics = data.get("topics", [])
        except json.JSONDecodeError:
            topics = [t.strip() for t in resp.strip().split(",") if t.strip()]

        return topics or ["general"]

    def _build_persona(self, topics: List[str]) -> Dict:
        prompt = f"""
Create a persona from these topics: {', '.join(topics)}

Return only valid JSON in the following format:
{{
  "inferred_role": "...",
  "primary_topics": ["..."],
  "key_interlocutors": ["..."],
  "risk_indicators_to_watch_for": ["..."]
}}
""".strip()
        resp = chat_completion(
            messages=[{"role": "user", "content": prompt}], max_tokens=400, temperature=0.1
        )
        return json.loads(resp.strip().removeprefix("```json").removesuffix("```"))

    def _register(self, email: str, refresh_token: str, persona: Dict) -> bool:
        payload = {
            "email": email,
            "refresh_token": encrypt_token(refresh_token),
            "persona": persona,
        }
        try:
            r = requests.post(
                f"{settings.backend_url}/register", json=payload, timeout=20
            )
            r.raise_for_status()
            return True
        except Exception as e:
            logging.error("Registration failed: %s", e)
            return False

    def run(self):
        logging.info("Starting onboarding...")
        service, creds, email = self._auth()
        if self._already_registered(email):
            logging.warning("Already registered – abort.")
            return

        samples = self._fetch_samples(service)
        if not samples:
            logging.error("No samples found.")
            return

        topics = []
        for s in samples:
            topics.extend(self._extract_topics(s))
            time.sleep(2)  # stay under Groq RPM/TPM

        if not topics:
            logging.error("No topics extracted.")
            return

        persona = self._build_persona(topics)
        logging.info("Persona:\n%s", json.dumps(persona, indent=2))

        if self._register(email, creds.refresh_token, persona):
            logging.info("✅ Onboarding complete for %s", email)
        else:
            logging.error("❌ Registration failed")


if __name__ == "__main__":
    PersonaOnboardingPlugin().run()