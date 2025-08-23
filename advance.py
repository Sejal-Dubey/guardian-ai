import asyncio
import websockets
import requests
import os
import json
import time
import random
from threading import Thread
from dotenv import load_dotenv

# --- Load Environment Variables ---
load_dotenv()

# ==============================================================================
# --- EMAIL SIMULATION LIBRARY ---
# ==============================================================================

EMAIL_SCENARIOS = [
    {
        "type": "Safe",
        "payload": {
            "from": '"Your Bank" <notifications@your-actual-bank.com>',
            "subject": "Your monthly statement is ready",
            "snippet": "Hi, your statement for the period ending August 2025 is now available. No action is required. Thank you for banking with us.",
            "auth": "dkim=pass; spf=pass; dmarc=pass (p=reject);"
        }
    },
    {
        "type": "Attack: Spear Phishing",
        "payload": {
            "from": '"HR Department" <hr@yourcompanny.com>', # Note the typo in the domain
            "subject": "URGENT: Mandatory New Benefits Enrollment",
            "snippet": "All employees are required to complete the new benefits enrollment by 5 PM today. Failure to comply will result in a lapse of coverage. Please log in to the new portal here immediately: http://yourcompanny-benefits-portal.com/login",
            "auth": "dkim=fail header.i=@yourcompanny.com; spf=softfail;"
        }
    },
    {
        "type": "Safe",
        "payload": {
            "from": '"Google Calendar" <calendar-notification@google.com>',
            "subject": "Invitation: Q4 Planning Session @ Wed Sep 3, 2025 10am - 11am",
            "snippet": "You have been invited to the following event. More details are available in your Google Calendar.",
            "auth": "dkim=pass; spf=pass;"
        }
    },
    {
        "type": "Attack: Credential Harvesting",
        "payload": {
            "from": '"IT Service Desk" <help@service-deskk.com>', # Extra 'k' in domain
            "subject": "Action Required: Your Mailbox is Almost Full",
            "snippet": "Your corporate mailbox is at 98% capacity. To avoid service disruption, you must upgrade your storage quota. Click here to authenticate and request more space: https://outlook.office365.com.service-deskk.com/increase-quota",
            "auth": "dkim=pass; spf=pass;" # Attacker's domain is valid, but impersonating
        }
    }
]


# ==============================================================================
# --- WebSocket Client ---
# ==============================================================================

async def listen_for_alerts():
    """Connects to the WebSocket and prints incoming alert messages."""
    # NOTE: Make sure your .env file has WS_AUTH_TOKEN="s3cr3t" or your chosen token
    auth_token = os.getenv("WS_AUTH_TOKEN", "s3cr3t")

    uri = f"ws://127.0.0.1:8000/ws/alerts?token={auth_token}"
    print(f"\n--- [WebSocket] Connecting to {uri} ---")
    try:
        async with websockets.connect(uri) as websocket:
            print("--- [WebSocket] ✅ Successfully connected. Monitoring for threats... ---")
            while True:
                message = await websocket.recv()
                alert_data = json.loads(message)
                
                # --- THIS IS THE UPDATE ---
                user_summary = alert_data.get("alert", {}).get("user_summary", "No summary available.")
                
                print("\n\n" + "#"*60)
                print("## 🚨 [WebSocket] INCOMING THREAT ALERT! 🚨 ##")
                print("#"*60)
                print(f"## User-Friendly Summary: {user_summary}")
                print("#"*60)
                print("## Full Alert Data:")
                print(json.dumps(alert_data, indent=2))
                print("#"*60 + "\n")

    except Exception as e:
        print(f"--- [WebSocket] ❌ Connection failed or dropped: {e} ---")

def run_listener_in_thread():
    """Runs the asyncio WebSocket listener in a separate thread."""
    asyncio.run(listen_for_alerts())


# ==============================================================================
# --- Main Simulation Runner ---
# ==============================================================================

def run_inbox_simulation(email: str):
    """Loops through scenarios and sends them to the backend for analysis."""
    print("\n--- [Simulator] Starting Real-Time Inbox Simulation ---")
    print(f"--- [Simulator] Monitoring inbox for: {email} ---")
    print("-" * 55)

    for scenario in EMAIL_SCENARIOS:
        delay = random.uniform(5, 10)
        print(f"\n--- [Simulator] Next email arriving in {int(delay)} seconds... ---")
        time.sleep(delay)

        email_type = scenario["type"]
        payload = scenario["payload"]
        
        print(f"--- [Simulator] 📥 Email Received! Type: '{email_type}' ---")
        print(f"    From: {payload['from']}")
        print(f"    Subject: {payload['subject']}")

        try:
            response = requests.post(
                "http://127.0.0.1:8000/debug/analyze-payload",
                json={"email": email, "email_data": payload},
                timeout=15
            )
            response.raise_for_status()
            print("--- [Simulator] ✅ Email forwarded to backend for analysis. ---")
        except requests.exceptions.RequestException as e:
            print(f"--- [Simulator] ❌ Error forwarding email to backend: {e} ---")
    
    print("\n\n" + "="*55)
    print("--- [Simulator] All simulated emails have been processed. ---")
    print("--- [Simulator] WebSocket listener will remain active. Press CTRL+C to exit. ---")
    print("="*55)


if __name__ == "__main__":
    registered_email = input("Enter the email address registered with the service: ")
    if not registered_email:
        print("Email cannot be empty.")
    else:
        listener_thread = Thread(target=run_listener_in_thread, daemon=True)
        listener_thread.start()
        time.sleep(3)
        run_inbox_simulation(registered_email)
        while listener_thread.is_alive():
            time.sleep(1)