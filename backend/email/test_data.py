from faker import Faker
import pandas as pd
import numpy as np

fake = Faker()

def generate_test_emails():
    """Generate realistic fraudulent emails for testing"""
    test_emails = []
    
    # Realistic fraud emails found in the wild
    realistic_fraud_emails = [
        {
            "sender": "security@paypal-security.com",
            "subject": "Security Alert: Unusual Activity Detected",
            "body": """
Dear Customer,

We've detected suspicious activity on your PayPal account. To protect your account, 
we've temporarily restricted access until verification is completed.

Please verify your identity at: https://secure.paypal.com/verify-login

If this wasn't you, contact our security team immediately.

This action is required within 24 hours to maintain account access.

PayPal Security Team
            """,
            "label": 1
        },
        {
            "sender": "billing@microsoft-updates.com",
            "subject": "Invoice #INV-2023-7897 - Past Due Payment Notice",
            "body": """
Dear Customer,

This is a reminder that your payment for invoice #INV-2023-7897 for $299.99 is past due.
To avoid service interruption, please update your payment information.

Our records indicate the following invoice is outstanding:
Invoice #: INV-2023-7897
Amount Due: $299.99
Due Date: 2023-11-15

Please click here to update your payment method:
https://billing.microsoft-updates.com/update-payment

If you believe this is an error, please contact our billing department.

Thank you for your prompt attention to this matter.

Microsoft Billing Department
            """,
            "label": 1
        },
        {
            "sender": "hr@amazon-corp.com",
            "subject": "Urgent: Your Amazon Account Suspension Notice",
            "body": """
Dear Valued Customer,

We've noticed violations of our policies on your Amazon account. As a result, 
we've suspended your account until further notice.

To appeal this suspension, you must complete the verification process immediately:
https://verify.account-status.com/amazon

Failure to complete verification within 72 hours will result in permanent account closure.

Amazon Account Compliance Team
            """,
            "label": 1
        },
        {
            "sender": "support@chase-alerts.com",
            "subject": "Security Alert: New Device Login Detected",
            "body": """
Dear Chase Customer,

We've detected a login to your Chase account from a new device or browser.

If this was you, no action is needed.

If this wasn't you, please secure your account immediately at:
https://secure.chase.com/security

Chase Account Security
            """,
            "label": 1
        },
        {
            "sender": "admin@your-bank-online.com",
            "subject": "Required Action: Verify Your Banking Information",
            "body": """
Dear Customer,

As part of our routine security upgrade, we require you to verify your banking information 
to ensure continued access to your account.

Please verify your details at: https://secure.your-bank-online.com/verify

This verification is required within 24 hours.

Thank you for your cooperation.

Your Bank Security Team
            """,
            "label": 1
        }
    ]
    
    # Generate test emails
    for idx, email in enumerate(realistic_fraud_emails):
        # Extract features
        num_links = len(email["body"].split('https://')) - 1 + len(email["body"].split('http://')) - 1
        num_dollar_signs = email["body"].count('$')
        num_urgent_terms = len([term for term in ['urgent', 'immediately', 'action', 'required', 'alert', 'verify', 'confirm'] 
                              if term.lower() in email["body"].lower()])
        num_personal_pronouns = len([pron for pron in ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'] 
                                    if pron.lower() in email["body"].lower()])
        
        # Domain match (will be 0 since these are deceptive domains)
        domain_match = 0
        
        # Suspicious subject
        suspicious_subject = 1 if any(word in email["subject"].lower() 
                                     for word in ['urgent', 'security alert', 'suspension', 'action required', 'verify']) else 0
        
        # Signature mismatch (these typically don't have proper signatures)
        signature_mismatch = 1
        
        test_emails.append([
            email["sender"],
            email["subject"],
            email["body"],
            0,  # DMARC/SPF pass (fail for these deceptive domains)
            email["label"],
            num_links,
            num_dollar_signs,
            num_urgent_terms,
            num_personal_pronouns,
            domain_match,
            suspicious_subject,
            signature_mismatch
        ])
    
    # Also add some legitimate test emails
    legitimate_emails = [
        {
            "sender": "john.smith@company.com",
            "subject": "Following up on our meeting",
            "body": """
Hi Sarah,

Great connecting with you today during our meeting about the Q4 marketing strategy. 
As discussed, I've attached the presentation materials for your review.

Let me know if you have any questions or need additional information.

We're excited about moving forward with these initiatives and look forward to our next check-in next week.

Best regards,
John Smith
Marketing Manager
Company
            """,
            "label": 0
        },
        {
            "sender": "billing@company.com",
            "subject": "Your November 2023 Statement",
            "body": """
Dear Customer,

Your November 2023 statement is now available in your secure online portal. You can 
access it by logging into your account at https://portal.company.com/statements.

The statement includes all transactions processed between November 1, 2023 and November 30, 2023.

If you have any questions about your statement, please don't hesitate to contact 
our customer service team at support@company.com or call 1-800-555-1234.

Thank you for your business.

Sincerely,
Company Billing Department
            """,
            "label": 0
        }
    ]
    
    for idx, email in enumerate(legitimate_emails):
        # Extract features
        num_links = len(email["body"].split('https://')) - 1 + len(email["body"].split('http://')) - 1
        num_dollar_signs = email["body"].count('$')
        num_urgent_terms = len([term for term in ['urgent', 'immediately', 'action', 'required', 'alert', 'verify', 'confirm'] 
                              if term.lower() in email["body"].lower()])
        num_personal_pronouns = len([pron for pron in ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'] 
                                    if pron.lower() in email["body"].lower()])
        
        # Domain match (will be 1 since these are legitimate domains)
        domain_match = 1
        
        # Suspicious subject
        suspicious_subject = 0
        
        # Signature mismatch (these have proper signatures)
        signature_mismatch = 0
        
        test_emails.append([
            email["sender"],
            email["subject"],
            email["body"],
            1,  # DMARC/SPF pass (pass for legitimate domains)
            email["label"],
            num_links,
            num_dollar_signs,
            num_urgent_terms,
            num_personal_pronouns,
            domain_match,
            suspicious_subject,
            signature_mismatch
        ])
    
    # Create DataFrame
    columns = [
        "sender", 
        "subject", 
        "body", 
        "dmarc_spf_pass", 
        "label",
        "num_links",
        "num_dollar_signs",
        "num_urgent_terms",
        "num_personal_pronouns",
        "domain_match",
        "suspicious_subject",
        "signature_mismatch"
    ]
    
    df = pd.DataFrame(test_emails, columns=columns)
    
    return df

# Save test data
test_df = generate_test_emails()
test_df.to_csv("test_email_fraud_dataset.csv", index=False)
print("Test data saved as 'test_email_fraud_dataset.csv'")
print(f"Test dataset contains {len(test_df)} samples with {test_df['label'].sum()} fraudulent emails")