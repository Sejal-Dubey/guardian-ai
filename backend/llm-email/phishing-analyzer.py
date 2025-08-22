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
    def __init__(self, api_key: str):
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

    def check_dkim(self, domain: str) -> Tuple[bool, Optional[str]]:
        """Check if a domain has a valid DKIM record."""
        if not domain: 
            return False, "No domain found"
        try:
            # Try common selectors: 'default', 'google', 'dkim'
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

    def calculate_dns_score(self, spf_status: bool, dkim_status: bool, dmarc_status: bool) -> float:
        """
        Calculate a more nuanced DNS authentication score that reflects actual security implications.
        
        We use a tiered system because:
        - All three passing is ideal but rare, so minimal risk
        - DKIM failure is particularly critical as it indicates potential content tampering
        - SPF and DMARC failures are concerning but less critical than DKIM
        
        Returns a score between 0.0 (minimal risk) and 1.0 (maximum risk)
        """
        # All three pass: minimal risk
        if spf_status and dkim_status and dmarc_status:
            return 0.1
        
        # DKIM fails but SPF and DMARC pass: high risk (DKIM is critical for email integrity)
        if dkim_status is False and spf_status and dmarc_status:
            return 0.8
        
        # Two pass, one fails (including DKIM): moderate risk
        if sum([spf_status, dkim_status, dmarc_status]) == 2:
            # If DKIM is one of the passing ones, risk is moderate
            if dkim_status:
                return 0.4
            # If DKIM failed, risk is higher
            else:
                return 0.6
        
        # Only one passes: high risk
        if sum([spf_status, dkim_status, dmarc_status]) == 1:
            # If DKIM is the one that passed, risk is moderate-high
            if dkim_status:
                return 0.5
            else:
                return 0.8
        
        # None pass: maximum risk
        return 0.9

    def check_link_domain_consistency(self, email_data: Dict, links: list) -> Dict:
        """Check if link domains match the sender domain."""
        sender_domain = email_data["domain"]
        
        consistent_links = 0
        inconsistent_links = []
        
        for link in links:
            # Extract domain from URL
            link_domain = link.split('//')[-1].split('/')[0].lower()
            
            # Check if sender domain is in link domain (for subdomains)
            if sender_domain in link_domain:
                consistent_links += 1
            else:
                inconsistent_links.append(link)
        
        consistency_ratio = consistent_links / max(1, len(links))
        
        return {
            "consistent_links": consistent_links,
            "inconsistent_links": inconsistent_links,
            "consistency_ratio": consistency_ratio,
            "risk_score": min(1.0, 1.0 - consistency_ratio)  # Higher inconsistency = higher risk
        }

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
        
        # Perform DNS authentication checks
        spf_status, spf_record = self.check_spf(email_data["domain"])
        dkim_status, dkim_record = self.check_dkim(email_data["domain"])
        dmarc_status, dmarc_record = self.check_dmarc(email_data["domain"])
        
        # Check link domain consistency
        link_consistency = self.check_link_domain_consistency(email_data, email_data["links"])
        
        # Cybersecurity Email Analysis Prompt with DNS authentication results
        prompt = f"""
You are a cybersecurity expert analyzing emails for potential threats. This analysis is for defensive security purposes only.

Email details:
- Sender Name: "{email_data['display_name']}"
- Sender Email: "{email_data['sender_email']}"
- Domain: "{email_data['domain']}"
- Subject: "{email_data['subject']}"
- Links: {email_data['links']}
- Body:
{email_data['body']}

Authentication Results:
- SPF: {"Valid" if spf_status else "Invalid/Not Found"}
- DKIM: {"Valid" if dkim_status else "Invalid/Not Found"}
- DMARC: {"Valid" if dmarc_status else "Invalid/Not Found"}

Link Domain Consistency:
- Consistent links: {link_consistency['consistent_links']}
- Inconsistent links: {link_consistency['inconsistent_links']}
- Consistency ratio: {link_consistency['consistency_ratio']:.2f}

Note: DNS authentication is critical. SPF verifies the sender's IP, DKIM verifies the email hasn't been tampered with, and DMARC specifies how to handle emails that fail SPF/DKIM. DKIM is particularly important because it protects against content modification.

Please analyze the following aspects:
1. Header analysis: Check for domain mismatches, unusual formatting, or signs of spoofing
2. Authorship verification: Verify if sender email matches displayed organization
3. Content evaluation: Look for unusual urgency, requests for sensitive information, or suspicious links

For financial institutions, legitimate security communications often contain urgency language. Focus on whether the email is requesting sensitive information or directing to suspicious websites.

Provide your analysis in this JSON format:
{{
    "header_risk_score": <float between 0.0-1.0>,
    "header_justification": "Explanation of header analysis factors",
    "authorship_risk_score": <float between 0.0-1.0>,
    "authorship_justification": "Explanation of authorship verification factors",
    "content_risk_score": <float between 0.0-1.0>,
    "content_justification": "Explanation of content evaluation factors",
    "final_risk_score": <float between 0.0-1.0>,
    "final_justification": "Explanation of how all aspects contributed to final score",
    "risk_level": "Low", "Medium", or "High",
    "indicators": ["indicator 1", "indicator 2"],
    "recommendation": "Specific action advice based on the analysis"
}}

Provide detailed justifications for each score to ensure explainable AI, highlighting the specific factors that led to the risk assessment.
"""
        try:
            response = self.model.generate_content(prompt)
            
            # Check if response was blocked by safety filters
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                return {
                    "header_risk_score": 0.5,
                    "header_justification": "AI analysis blocked by safety filters",
                    "authorship_risk_score": 0.5,
                    "authorship_justification": "AI analysis blocked by safety filters",
                    "content_risk_score": 0.5,
                    "content_justification": "AI analysis blocked by safety filters",
                    "final_risk_score": 0.5,
                    "final_justification": "AI analysis blocked by safety filters",
                    "risk_level": "Medium",
                    "indicators": ["AI analysis blocked"],
                    "recommendation": "Unable to analyze automatically. Please review manually."
                }
            
            # Extract and clean the JSON response
            cleaned_response = response.text.strip()
            
            # Remove markdown formatting if present
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            elif cleaned_response.startswith("json"):
                cleaned_response = cleaned_response[4:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
                
            return json.loads(cleaned_response)
                
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return {
                "header_risk_score": 0.5,
                "header_justification": "JSON parsing failed",
                "authorship_risk_score": 0.5,
                "authorship_justification": "JSON parsing failed",
                "content_risk_score": 0.5,
                "content_justification": "JSON parsing failed",
                "final_risk_score": 0.5,
                "final_justification": "JSON parsing failed",
                "risk_level": "Medium",
                "indicators": ["JSON parsing failed"],
                "recommendation": "Unable to parse AI response. Please manually review this email."
            }
        except Exception as e:
            print(f"Analysis error: {e}")
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
        
        # Extract links from the body
        email_data["links"] = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', email_data["body"])
        
        # Perform DNS authentication checks
        spf_status, _ = self.check_spf(email_data["domain"])
        dkim_status, _ = self.check_dkim(email_data["domain"])
        dmarc_status, _ = self.check_dmarc(email_data["domain"])
        
        # Calculate DNS-based validation score using the new tiered system
        dns_score = self.calculate_dns_score(spf_status, dkim_status, dmarc_status)
        
        # Check link domain consistency
        link_consistency = self.check_link_domain_consistency(email_data, email_data["links"])
        
        # Analyze content using LLM
        content_analysis = self.analyze_content(email_data)
        
        # Get the final content analysis score
        final_content_score = content_analysis.get("final_risk_score", 0.5)
        
        # Calculate overall risk score with DNS validation as a factor
        # Weighted average: 30% DNS checks, 20% link consistency, 50% content analysis
        final_score = (0.3 * dns_score) + (0.2 * link_consistency["risk_score"]) + (0.5 * final_content_score)
        
        # Determine risk level based on final_score
        risk_level = "High" if final_score > 0.7 else "Medium" if final_score > 0.4 else "Low"
        
        # Determine analysis confidence
        # Check if content_analysis has the expected fields
        is_llm_analysis = all(key in content_analysis for key in [
            "header_risk_score", "authorship_risk_score", "content_risk_score", 
            "final_risk_score", "indicators", "recommendation"
        ])
        
        analysis_confidence = "High" if is_llm_analysis else "Medium"
        
        return {
            "final_score": round(final_score, 2),
            "risk_level": risk_level,
            "analysis_confidence": analysis_confidence,
            "dns_authentication": {
                "spf_valid": spf_status,
                "dkim_valid": dkim_status,
                "dmarc_valid": dmarc_status,
                "dns_score": dns_score,
                "dns_justification": self.get_dns_justification(spf_status, dkim_status, dmarc_status)
            },
            "link_consistency": link_consistency,
            "details": {
                "content_analysis": content_analysis,
                "parsed_email": email_data
            }
        }
    
    def get_dns_justification(self, spf_status: bool, dkim_status: bool, dmarc_status: bool) -> str:
        """Generate a justification for the DNS authentication score."""
        if spf_status and dkim_status and dmarc_status:
            return "All authentication methods passed (SPF, DKIM, DMARC), indicating a highly secure email."
        
        if not dkim_status and spf_status and dmarc_status:
            return "DKIM authentication failed while SPF and DMARC passed. DKIM is critical for verifying email integrity, so this presents a significant security risk."
        
        if sum([spf_status, dkim_status, dmarc_status]) == 2:
            failed_method = "SPF" if not spf_status else ("DKIM" if not dkim_status else "DMARC")
            return f"Two authentication methods passed, but {failed_method} failed. This represents a moderate security risk."
        
        if sum([spf_status, dkim_status, dmarc_status]) == 1:
            passed_method = "SPF" if spf_status else ("DKIM" if dkim_status else "DMARC")
            return f"Only {passed_method} authentication passed, indicating a high security risk."
        
        return "No authentication methods passed (SPF, DKIM, DMARC all failed), indicating maximum security risk."

# --- Example Usage ---
if __name__ == "__main__":
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

You can access your statement by logging into your account at https://www.kotakmahindra.com/offers/login  and navigating to the 'Statements' section.

For assistance, please contact our customer care at 1860 266 2666 or visit your nearest Kotak branch.

Thank you for banking with Kotak Bank.

Best regards,
Kotak Bank Team
"""
        
        print("=== Analyzing Bank Email ===")
        result = detector.score_email(legitimate_email)
        print(json.dumps(result, indent=2))