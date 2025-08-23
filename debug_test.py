#!/usr/bin/env python3
"""
Debug script for Email Security System
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"
EMAIL = "sejaldubey903@gmail.com"

def test_system():
    print("=" * 60)
    print("EMAIL SECURITY SYSTEM DIAGNOSTICS")
    print("=" * 60)
    
    # 1. Check registration status
    print("\n1. Checking registration status...")
    try:
        resp = requests.get(f"{BASE_URL}/status/{EMAIL}")
        if resp.status_code == 200:
            print(f"✅ User {EMAIL} is registered")
        else:
            print(f"❌ User not registered: {resp.text}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return
    
    # 2. Check webhook configuration
    print("\n2. Checking Gmail webhook configuration...")
    try:
        resp = requests.get(f"{BASE_URL}/debug/webhook-status/{EMAIL}")
        data = resp.json()
        if data.get("webhook_configured"):
            print(f"✅ Webhook configured (History ID: {data.get('history_id')})")
        else:
            print("❌ Webhook not configured")
        print(f"   Last refresh: {data.get('last_refresh')}")
    except Exception as e:
        print(f"⚠️  Debug endpoint not available: {e}")
    
    # 3. Test WebSocket alert
    print("\n3. Testing WebSocket alerts...")
    print("   Sending test alert to all connected clients...")
    try:
        resp = requests.post(f"{BASE_URL}/debug/send-test-alert")
        data = resp.json()
        print(f"✅ Alert sent to {data.get('connected_clients', 0)} clients")
        if data.get('connected_clients', 0) == 0:
            print("   ⚠️  No WebSocket clients connected!")
    except Exception as e:
        print(f"⚠️  Could not send test alert: {e}")
    
    # 4. Check recent emails
    print("\n4. Analyzing recent emails...")
    try:
        resp = requests.get(f"{BASE_URL}/debug/check-recent-emails/{EMAIL}")
        data = resp.json()
        
        if "error" in data:
            print(f"❌ Error: {data['error']}")
        else:
            emails = data.get("recent_emails", [])
            print(f"   Found {len(emails)} recent emails")
            
            for i, email_info in enumerate(emails, 1):
                email = email_info["email"]
                analysis = email_info["analysis"]
                
                print(f"\n   Email {i}:")
                print(f"   From: {email['from'][:50]}...")
                print(f"   Subject: {email['subject'][:50]}...")
                print(f"   Date: {email.get('date', 'Unknown')}")
                
                if "error" in analysis:
                    print(f"   ❌ Analysis failed: {analysis['error']}")
                else:
                    threat = "🚨 THREAT" if analysis.get("is_threat") else "✅ SAFE"
                    print(f"   Analysis: {threat}")
                    if analysis.get("is_threat"):
                        print(f"   Type: {analysis.get('threat_type')}")
                        print(f"   Confidence: {analysis.get('confidence', 0)*100:.1f}%")
                        print(f"   Reason: {analysis.get('justification')}")
                        print(f"   Action: {analysis.get('action')}")
    except Exception as e:
        print(f"⚠️  Could not check recent emails: {e}")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 60)

def send_suspicious_test_email():
    """Instructions for sending a test phishing email"""
    print("\n" + "=" * 60)
    print("TEST EMAIL TEMPLATES")
    print("=" * 60)
    print("\nSend one of these test emails to yourself to trigger alerts:\n")
    
    templates = [
        {
            "subject": "URGENT: Your account will be suspended",
            "body": "Dear user, click here to verify your account: http://bit.ly/verify-account"
        },
        {
            "subject": "You've won $1,000,000!",
            "body": "Congratulations! You've been selected. Click here to claim: http://win-prize.tk"
        },
        {
            "subject": "Invoice #12345 - Payment Required",
            "body": "Please pay the attached invoice immediately to avoid legal action. Download: http://invoice-pay.ml/doc.exe"
        },
        {
            "subject": "Security Alert: Unauthorized Access Detected",
            "body": "We detected unusual activity. Reset your password here: http://amaz0n.com/security"
        }
    ]
    
    for i, template in enumerate(templates, 1):
        print(f"\nTemplate {i}:")
        print(f"Subject: {template['subject']}")
        print(f"Body: {template['body']}")
    
    print("\n⚠️  These are for testing only in your own Gmail account!")

if __name__ == "__main__":
    import sys
    
    print("Email Security System Debug Tool\n")
    print("1. Run diagnostics")
    print("2. Show test email templates")
    print("3. Both")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == "1":
        test_system()
    elif choice == "2":
        send_suspicious_test_email()
    elif choice == "3":
        test_system()
        send_suspicious_test_email()
    else:
        print("Invalid choice")