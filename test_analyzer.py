import asyncio
import json
import logging
import sys
import os
from unittest.mock import patch, MagicMock

# --- Self-aware path correction ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
except NameError:
    script_dir = os.getcwd()
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] test_analyzer: %(message)s",
)

try:
    from backend.backend import analyze, USER_DB
except ImportError:
    print("FATAL ERROR: Could not import from 'backend'. Please ensure the 'backend' folder exists in the same directory as this script.")
    exit(1)

# --- Base Sophisticated Attack Scenarios ---
test_service_hijack = {
    "from": "Asana <notifications@asana.com>",
    "subject": "John Doe has assigned you a task: 'Review Q3 Financials'",
    "snippet": "Hi, Please review the attached Q3 financial document and provide your feedback by EOD. The document can be accessed here: https://asana-document-portal.com/files/Q3-Financials.pdf",
    "auth": "mx.google.com; spf=pass; dkim=pass; dmarc=pass",
}
test_ceo_impersonation = {
    "from": "Your CEO <ceo@myc0mpany.com>",
    "subject": "Following up from our chat at the conference",
    "snippet": "Great seeing you at the event last week. As we discussed, I've put together the draft for the new incentive plan. Can you give it a quick review on your phone? Need your thoughts before the board meeting tomorrow. Link: http://myc0mpany.com/drafts/incentive-plan.docx",
    "auth": "mx.google.com; spf=pass; dkim=pass; dmarc=pass",
}
test_hr_scare = {
    "from": "HR Department <hr@yourcompany.com>",
    "subject": "Action Required: Discrepancy in Your Payroll Information",
    "snippet": "Our system has detected a mismatch in your payroll details. To ensure you receive your next payment on time, please log in to our secure portal at https://hr.yourcompany.com/payroll and verify your information immediately.",
    "auth": "mx.google.com; spf=softfail; dkim=pass; dmarc=pass",
}
test_benign_complex_email = {
    "from": "GitHub <noreply@github.com>",
    "subject": "[your-repo] Alert: High severity vulnerability found in 'requests'",
    "snippet": "Dependabot has detected a high severity vulnerability in 'requests'. We recommend you upgrade to version 2.28.1 or later. You can view the full alert here: https://github.com/your-repo/security/dependabot/1",
    "auth": "mx.google.com; spf=pass; dkim=pass; dmarc=pass",
}

# --- NEW Black Hat Level Scenarios ---

# Test Case 5: The "Personal Email & Brand Spoof Combo"
# An attacker spoofs a known personal contact (bhujbalsai@gmail.com) and uses the name of a trusted brand (Gemini)
# to create a highly convincing lure. The technical failure (spf=fail) is the primary giveaway that a sophisticated
# AI should catch immediately, despite the compelling social context.
test_personal_brand_spoof = {
    "from": "Sai Bhujbal via Gemini <bhujbalsai@gmail.com>",
    "subject": "Your Gemini API Usage & New Project Idea",
    "snippet": "Hey, it's Sai. Was looking over your recent Gemini API usage and had an idea for a project we could collaborate on. I've put the notes in a shared doc, take a look and let me know your thoughts: https://docs.google.com/document/d/1a2b3c4d5e_FAKE_ID/edit?usp=sharing_link_impersonation",
    "auth": "mx.google.com; spf=fail (google.com does not designate sender IP as permitted sender); dkim=neutral (no key for signature); dmarc=fail (p=REJECT)",
}

# Test Case 6: The "Indian Bank Geo-Targeted Phish" (HDFC)
# This is a geo-targeted attack that uses a well-known Indian bank (HDFC) and references a real service (NetBanking).
# It uses a "cyber squatting" domain (hdfc-alerts.in) which is designed to look official. The urgency around a
# "high-value transaction" is a powerful psychological trigger.
test_hdfc_bank_phish = {
    "from": "HDFC Bank Alerts <alerts@hdfc-alerts.in>",
    "subject": "Security Alert: High-value transaction processed from your NetBanking account",
    "snippet": "Dear Customer, a transaction of INR 45,000.00 has been processed from your account. If you did not authorize this, please login immediately to our new security portal to verify and cancel the transaction: https://netbanking.hdfc-alerts.in/verify/transaction_id=89a4b1c",
    "auth": "mx.google.com; spf=pass; dkim=pass; dmarc=pass",
}

# Test Case 7: The "OpenAI Security Scare"
# This attack leverages the authority and technical nature of OpenAI. It uses a subdomain lure
# (security.accounts.openai.net) that looks extremely plausible. The mention of "unusual API activity from a foreign IP"
# is designed to panic a developer or user into immediate action.
test_openai_scare = {
    "from": "OpenAI Security Team <noreply@openai.com>",
    "subject": "Action Required: Unusual API Activity Detected on Your Account",
    "snippet": "We have detected potentially malicious activity on your OpenAI account originating from an IP in Eastern Europe. To protect your account, we have temporarily suspended your API keys. Please review the activity and re-enable your keys at our security center: https://security.accounts.openai.net/alerts/review",
    "auth": "mx.google.com; spf=pass; dkim=pass; dmarc=pass",
}

async def run_tests():
    test_user_email = "sejaldubey903@gmail.com"
    if test_user_email not in USER_DB:
        print(f"Error: Test user '{test_user_email}' not found in user_database.json.")
        return

    tests = {
        "Sophisticated Service Hijack": test_service_hijack,
        "CEO Impersonation (Cousin Domain)": test_ceo_impersonation,
        "HR Payroll Scare (Ambiguous Auth)": test_hr_scare,
        "Benign but Complex Email (False Positive Check)": test_benign_complex_email,
        "--- ADVANCED ATTACKS ---": {},
        "Personal Email & Brand Spoof (Gemini)": test_personal_brand_spoof,
        "Indian Bank Geo-Targeted Phish (HDFC)": test_hdfc_bank_phish,
        "OpenAI API Security Scare": test_openai_scare,
    }

    print("--- Running Sophisticated Email Security Analysis Tests ---")

    with patch("backend.backend.mgr.broadcast") as mock_broadcast:
        for name, email_data in tests.items():
            if not email_data: # For the separator
                print(f"\n--- {name} ---")
                continue

            print(f"\n--- [TESTING] Scenario: {name} ---")
            mock_broadcast.reset_mock()
            await analyze(test_user_email, email_data)
            
            if mock_broadcast.called:
                call_args = mock_broadcast.call_args[0][0]
                alert_data = json.loads(call_args)
                print(f"  [RESULT] THREAT DETECTED  ✅")
                # --- KEY FIX HERE ---
                # Changed 'justification' to 'reasoning' to match the new prompt output
                print(f"    -> Reasoning: {alert_data['alert']['reasoning']}")
                print(f"    -> Action: {alert_data['alert']['action']}")
            else:
                print(f"  [RESULT] Email Classified as SAFE 🛡️")

if __name__ == "__main__":
    asyncio.run(run_tests())