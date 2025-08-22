import google.generativeai as genai
import dns.resolver
import email
from email.header import decode_header
import re
import json
import os
from typing import Dict, Tuple, Optional
import email.utils

class SpearPhishingDetector:
    def _init_(self, api_key: str):
        """Initialize the phishing detector with Gemini API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
    def check_spf(self, domain: str) -> Tuple[bool, Optional[str]]:
        """Check if a domain has a valid SPF record."""
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

    def check_dmarc(self, domain: str) -> Tuple[bool, Optional[str]]:
        """Check if a domain has a valid DMARC record."""
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

    def parse_email(self, raw_email: str) -> Dict:
        """Parse raw email and extract relevant components."""
        msg = email.message_from_string(raw_email)
        
        # Improved parsing of From header
        from_header = msg.get("From", "")
        sender = from_header
        
        # Extract display name and email address separately
        display_name, sender_email = email.utils.parseaddr(from_header)
        
        subject = msg.get("Subject", "")
        
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    try:
                        body += part.get_payload(decode=True).decode()
                    except:
                        body += str(part.get_payload())
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                try:
                    body = payload.decode()
                except:
                    body = str(msg.get_payload())
            else:
                body = str(msg.get_payload())
            
        # Extract domain from email address
        sender_domain = sender_email.split('@')[-1] if '@' in sender_email and sender_email else ""
            
        return {
            "sender": sender, 
            "display_name": display_name,
            "sender_email": sender_email,
            "domain": sender_domain,
            "subject": subject, 
            "body": body
        }

    def analyze_content(self, email_data: Dict) -> Dict:
        """Analyze email content for phishing indicators using LLM analysis."""
        # Extract links from the body
        email_data["links"] = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_data["body"])
        
        # Cybersecurity Email Analysis Prompt
        prompt = f"""
As a cybersecurity expert, analyze this email for potential threats across three critical dimensions: header analysis, authorship verification, and content evaluation. This assessment is for defensive security purposes only.

Email details:
- Sender Name: "{email_data['display_name']}"
- Sender Email: "{email_data['sender_email']}"
- Domain: "{email_data['domain']}"
- Subject: "{email_data['subject']}"
- Links: {email_data['links']}
- Body:
{email_data['body']}

For your analysis, consider these phishing indicators:
1. Domain impersonation (using similar but not identical domains)
2. Unusual urgency or pressure tactics
3. Requests for sensitive information
4. Suspicious links
5. Grammatical errors or awkward phrasing
6. Generic greetings instead of personalized ones
7. Mismatch between sender email and displayed name

For banks and financial institutions, legitimate security alerts often contain urgency language. Focus on whether the email is asking for sensitive information or directing to suspicious sites.

Return in this exact JSON format:
{{
    "header_risk_score": <float 0.0-1.0>,
    "header_justification": "Explanation of header analysis factors",
    "authorship_risk_score": <float 0.0-1.0>,
    "authorship_justification": "Explanation of authorship verification factors",
    "content_risk_score": <float 0.0-1.0>,
    "content_justification": "Explanation of content evaluation factors",
    "final_risk_score": <float 0.0-1.0>,
    "final_justification": "Explanation of how all aspects contributed to final score",
    "risk_level": "Low", "Medium", or "High",
    "indicators": ["indicator 1", "indicator 2"],
    "recommendation": "Specific action advice based on the analysis"
}}

Provide detailed justifications for each score to ensure explainable AI, highlighting the specific factors that led to the risk assessment.
"""
        try:
            response = self.model.generate_content(prompt)
            
            # Extract and clean the JSON response
            cleaned_response = response.text.strip()
            
            # Remove markdown formatting if present
            if cleaned_response.startswith("json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("json"):
                cleaned_response = cleaned_response[4:]
            if cleaned_response.endswith(""):
                cleaned_response = cleaned_response[:-3]
                
            return json.loads(cleaned_response)
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            # Return a basic analysis if JSON parsing fails
            return {
                "header_risk_score": 0.5,
                "header_justification": "Unable to parse AI response",
                "authorship_risk_score": 0.5,
                "authorship_justification": "Unable to parse AI response",
                "content_risk_score": 0.5,
                "content_justification": "Unable to parse AI response",
                "final_risk_score": 0.5,
                "final_justification": "Unable to parse AI response",
                "risk_level": "Medium",
                "indicators": ["Unable to parse AI response"],
                "recommendation": "Unable to parse AI response. Please manually review this email."
            }
        except Exception as e:
            print(f"Analysis error: {e}")
            # Return a basic analysis if something else fails
            return {
                "header_risk_score": 0.5,
                "header_justification": "Analysis failed",
                "authorship_risk_score": 0.5,
                "authorship_justification": "Analysis failed",
                "content_risk_score": 0.5,
                "content_justification": "Analysis failed",
                "final_risk_score": 0.5,
                "final_justification": "Analysis failed",
                "risk_level": "Medium",
                "indicators": ["Analysis failed"],
                "recommendation": "Analysis failed. Please manually review this email."
            }

    def score_email(self, raw_email: str) -> Dict:
        """Score an email for phishing indicators."""
        # Parse the email to extract components
        email_data = self.parse_email(raw_email)
        
        # Perform DNS checks
        spf_status, _ = self.check_spf(email_data["domain"])
        dmarc_status, _ = self.check_dmarc(email_data["domain"])
        
        # Analyze content using LLM
        content_analysis = self.analyze_content(email_data)
        
        # Calculate DNS-based validation score
        dns_validated = spf_status and dmarc_status
        dns_score = 0.1 if dns_validated else 0.7
        
        # Get the final content analysis score
        final_content_score = content_analysis.get("final_risk_score", 0.5)
        
        # Calculate overall risk score with DNS validation as a factor
        # Weighted average: 30% DNS checks, 70% content analysis
        final_score = (0.3 * dns_score) + (0.7 * final_content_score)
        
        # Determine risk level based on final_score
        risk_level = "High" if final_score > 0.7 else "Medium" if final_score > 0.4 else "Low"
        
        return {
            "final_score": round(final_score, 2),
            "risk_level": risk_level,
            "details": {
                "spf_valid": spf_status,
                "dmarc_valid": dmarc_status,
                "dns_score": dns_score,
                "content_analysis": content_analysis,
                "parsed_email": email_data
            }
        }

# --- Example Usage ---
if _name_ == "_main_":
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyDZ21jqGvrnsDF1ZMkm1JxzBsFQ5dedDns")
    
    if "YOUR_API_KEY" in gemini_api_key:
        print("WARNING: Please replace 'YOUR_API_KEY' with your actual Gemini API key.")
    else:
        detector = SpearPhishingDetector(gemini_api_key)
        
        # Test with the legitimate bank email
        legitimate_email = """From: Kotak Bank <security@kotak.com>
To: customer@example.com
Subject: Your Kotak Bank Monthly Statement - November 2023

Dear Customer,

Your monthly statement for November 2023 is now available in your Kotak Bank online account.

You can access your statement by logging into your account at https://www.kotakmahindra.com/offers/login'  and navigating to the 'Statements' section.

For assistance, please contact our customer care at 1860 266 2666 or visit your nearest Kotak branch.

Thank you for banking with Kotak Bank.

Best regards,
Kotak Bank Team
"""
        
        print("=== Analyzing Bank Email ===")
        result = detector.score_email(legitimate_email)
        print(json.dumps(result, indent=2))