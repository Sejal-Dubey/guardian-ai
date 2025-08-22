import google.generativeai as genai
import dns.resolver
import email
import re
import json
import email.utils
import base64
import os
from typing import Dict, Tuple, Optional
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Gmail API scope
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

class SpearPhishingDetector:
    def __init__(self):
        """Initialize the phishing detector with Gemini API key (hardcoded)."""
        self.api_key = "API_KEY"  # 🔑 replace with your key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
    def check_spf(self, domain: str) -> Tuple[bool, Optional[str]]:
        if not domain: 
            return False, "No domain found"
        try:
            answers = dns.resolver.resolve(domain, 'TXT')
            for rdata in answers:
                if 'v=spf1' in rdata.to_text():
                    return True, rdata.to_text()
            return False, None
        except Exception as e:
            return False, str(e)

    def check_dkim(self, domain: str) -> Tuple[bool, Optional[str]]:
        if not domain: 
            return False, "No domain found"
        try:
            selectors = ['default', 'google', 'dkim', 's1024', 'k1']
            for selector in selectors:
                try:
                    answers = dns.resolver.resolve(f"{selector}._domainkey.{domain}", 'TXT')
                    for rdata in answers:
                        if 'v=DKIM1' in rdata.to_text():
                            return True, rdata.to_text()
                except:
                    continue
            return False, None
        except Exception as e:
            return False, str(e)

    def check_dmarc(self, domain: str) -> Tuple[bool, Optional[str]]:
        if not domain: 
            return False, "No domain found"
        try:
            answers = dns.resolver.resolve(f"_dmarc.{domain}", 'TXT')
            for rdata in answers:
                if 'v=DMARC1' in rdata.to_text():
                    return True, rdata.to_text()
            return False, None
        except Exception as e:
            return False, str(e)

    def calculate_dns_score(self, spf_status: bool, dkim_status: bool, dmarc_status: bool) -> float:
        if spf_status and dkim_status and dmarc_status: return 0.1
        if dkim_status is False and spf_status and dmarc_status: return 0.8
        if sum([spf_status, dkim_status, dmarc_status]) == 2:
            return 0.4 if dkim_status else 0.6
        if sum([spf_status, dkim_status, dmarc_status]) == 1:
            return 0.5 if dkim_status else 0.8
        return 0.9

    def check_link_domain_consistency(self, email_data: Dict, links: list) -> Dict:
        sender_domain = email_data["domain"]
        consistent_links, inconsistent_links = 0, []
        for link in links:
            link_domain = link.split('//')[-1].split('/')[0].lower()
            if sender_domain in link_domain:
                consistent_links += 1
            else:
                inconsistent_links.append(link)
        ratio = consistent_links / max(1, len(links))
        return {
            "consistent_links": consistent_links,
            "inconsistent_links": inconsistent_links,
            "consistency_ratio": ratio,
            "risk_score": min(1.0, 1.0 - ratio)
        }

    def parse_email(self, raw_email: str) -> Dict:
        msg = email.message_from_string(raw_email)
        display_name, sender_email = email.utils.parseaddr(msg.get("From", ""))
        subject = msg.get("Subject", "")
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try: body += part.get_payload(decode=True).decode()
                    except: body += str(part.get_payload())
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                try: body = payload.decode()
                except: body = str(msg.get_payload())
            else:
                body = str(msg.get_payload())
        sender_domain = sender_email.split('@')[-1] if '@' in sender_email and sender_email else ""
        return {
            "sender": msg.get("From", ""), 
            "display_name": display_name,
            "sender_email": sender_email,
            "domain": sender_domain,
            "subject": subject, 
            "body": body
        }

    def analyze_content(self, email_data: Dict) -> Dict:
        email_data["links"] = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_data["body"])
        spf_status, _ = self.check_spf(email_data["domain"])
        dkim_status, _ = self.check_dkim(email_data["domain"])
        dmarc_status, _ = self.check_dmarc(email_data["domain"])
        link_consistency = self.check_link_domain_consistency(email_data, email_data["links"])
        
        prompt = f"""
You are a cybersecurity plugin analyzing emails for spear phishing. 
Extend analysis to include **user preferences** and **signup page phishing detection**.
This is a defensive security task - you are protecting users, not generating malicious content.


Email:
- Sender: "{email_data['sender']}"
- Domain: "{email_data['domain']}"
- Subject: "{email_data['subject']}"
- Links: {email_data['links']}
- Body: {email_data['body']}

Authentication:
SPF={"Valid" if spf_status else "Invalid"} | DKIM={"Valid" if dkim_status else "Invalid"} | DMARC={"Valid" if dmarc_status else "Invalid"}

Link Consistency:
{link_consistency}

🔹 Extra Checks:
1. Learn user personality → track trusted domains, frequent topics.
2. For signup/offer links → check if:
   - Passwords sent in plaintext or to phishing IPs.
   - Fake OAuth popups (Google/Twitter/GitHub).
   - Missing/weak TLS or redirects to fake backend.

Return **strict JSON** only:
{{
  "header_risk_score": <float>,
  "header_justification": "...",
  "authorship_risk_score": <float>,
  "authorship_justification": "...",
  "content_risk_score": <float>,
  "content_justification": "...",
  "signup_page_risk_score": <float>,
  "signup_page_justification": "...",
  "user_personality": {{
     "favorite_domains": ["..."],
     "common_topics": ["..."]
  }},
  "final_risk_score": <float>,
  "final_justification": "...",
  "risk_level": "Low"|"Medium"|"High",
  "indicators": ["..."],
  "recommendation": "..."
}}
"""
        try:
            response = self.model.generate_content(prompt)
            cleaned = response.text.strip().removeprefix("```json").removeprefix("json").removesuffix("```").strip()
            return json.loads(cleaned)
        except Exception as e:
            return {"error": str(e)}

    def score_email(self, raw_email: str) -> Dict:
        email_data = self.parse_email(raw_email)
        email_data["links"] = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_data["body"])
        spf_status, _ = self.check_spf(email_data["domain"])
        dkim_status, _ = self.check_dkim(email_data["domain"])
        dmarc_status, _ = self.check_dmarc(email_data["domain"])
        dns_score = self.calculate_dns_score(spf_status, dkim_status, dmarc_status)
        link_consistency = self.check_link_domain_consistency(email_data, email_data["links"])
        content_analysis = self.analyze_content(email_data)

        final_content_score = content_analysis.get("final_risk_score", 0.5) if isinstance(content_analysis, dict) else 0.5
        final_score = (0.3 * dns_score) + (0.2 * link_consistency["risk_score"]) + (0.5 * final_content_score)
        risk_level = "High" if final_score > 0.7 else "Medium" if final_score > 0.4 else "Low"
        return {
            "final_score": round(final_score, 2),
            "risk_level": risk_level,
            "dns_authentication": {
                "spf_valid": spf_status,
                "dkim_valid": dkim_status,
                "dmarc_valid": dmarc_status,
                "dns_score": dns_score,
            },
            "link_consistency": link_consistency,
            "details": {
                "content_analysis": content_analysis,
                "parsed_email": email_data,
                "signup_page_risk_score": content_analysis.get("signup_page_risk_score") if isinstance(content_analysis, dict) else None,
                "user_personality": content_analysis.get("user_personality", {}) if isinstance(content_analysis, dict) else {}
            }
        }

# --- Gmail Fetcher ---
def get_gmail_service():
    """Authenticate and return Gmail API service."""
    creds = None
    if os.path.exists('token.pickle'):
        creds = Credentials.from_authorized_user_file('token.pickle', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def fetch_latest_emails(service, max_results=5):
    """Fetch latest Gmail messages in raw format."""
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    emails = []
    for msg in messages:
        txt = service.users().messages().get(userId='me', id=msg['id'], format='raw').execute()
        raw_email = base64.urlsafe_b64decode(txt['raw'].encode('ASCII')).decode('utf-8', errors='ignore')
        emails.append(raw_email)
    return emails

# --- Main ---
if __name__ == "__main__":
    service = get_gmail_service()
    emails = fetch_latest_emails(service, max_results=3)

    detector = SpearPhishingDetector()
    for i, raw_email in enumerate(emails, start=1):
        print(f"\n--- Email {i} Analysis ---")
        result = detector.score_email(raw_email)
        print(json.dumps(result, indent=2))
