import pandas as pd
import numpy as np
import random
import re
from faker import Faker
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import warnings
warnings.filterwarnings('ignore')

# Initialize Faker
fake = Faker()

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_advanced_emails(num_samples):
    """
    Generates a more realistic synthetic dataset with balanced features and ambiguous samples.
    """
    emails = []
    
    # Fraud templates with legitimate-looking elements
    fraud_templates = [
        {
            "fraud_probability": 0.7,
            "subject_templates": [
                "Security Alert: {company} Account Activity",
                "Action Required: Update Payment Information",
                "Notification: {company} - Account Verification Needed",
                "Invoice {inv_num}: Past Due Payment Notice"
            ],
            "body_templates": [
                """
Dear {name},

We've detected suspicious activity on your {company} account. For your security, 
we've temporarily restricted access until verification is completed.

Please verify your identity at: https://secure.{company}.com/verify?token={uuid}

If this wasn't you, contact security immediately: security@{company}.com

Sincerely,
{company} Security Team
                """,
                """
Hi {name},

This is a reminder that your payment for invoice #{inv_num} for ${amount} is past due.
To avoid service interruption, please update your payment information.

Please click here to update your payment method: 
https://billing.{company}.com/update-payment?token={uuid}

If you believe this is an error, please contact our billing department.

Best regards,
{company} Billing Department
                """,
                """
Hello {name},

Following up on our conversation about the {project} project. As discussed, 
we've updated our banking information due to a system upgrade:

New Payment Details:
Account Name: {company} Services
Account #: {account_number}
Routing #: {routing_number}

Please direct all future payments to this new account.

Sincerely,
{name}
{position}
{company}
                """
            ]
        }
    ]
    
    # Legitimate templates with business context
    legitimate_templates = [
        {
            "subject_templates": [
                "Meeting Follow-up: {topic}",
                "Your {month} {year} Statement",
                "Project Update: {project_name}",
                "Welcome {name} to {company}!"
            ],
            "body_templates": [
                """
Hi {name},

Great connecting with you during our meeting about {topic}. As discussed, 
I've attached the presentation materials for your review.

We're excited about our collaboration and look forward to our next check-in on {follow_up_date}.

Best regards,
{name}
{position}
{company}
                """,
                """
Dear {name},

Your {month} {year} statement is now available in your secure portal. You can 
access it by logging into your account at https://portal.{company}.com/statements.

If you have any questions about your statement, please reply to this email.

Thank you for your business.

Sincerely,
{company_name} Billing Department
                """,
                """
Hello Team,

Quick update on the {project_name} project. Milestones are on track. 
We've completed {percentage}% of the deliverables ahead of schedule.

Meeting scheduled for {meeting_date} to discuss next steps.

Best regards,
{name}
{position}
{company}
                """
            ]
        }
    ]
    
    # Generate email samples
    for i in range(num_samples):
        # Determine email type (fraud or legitimate) with 50% balance
        email_type = random.choices(
            ['fraud', 'legitimate'],
            weights=[0.5, 0.5],  # Equal balance
            k=1
        )[0]
        
        # Generate common elements
        name = fake.first_name()
        last_name = fake.last_name()
        company = fake.company()
        company_short = company.lower().replace(' ', '').replace(',', '').replace('.', '').replace('&', 'and')[:20]
        domain = f"{company_short}.com"
        
        # Generate a random invoice number that might be used multiple times
        invoice_number = fake.random_int(min=1000, max=9999)
        
        if email_type == 'fraud':
            # Create potentially deceptive sender domains
            sender_options = [
                f"security@{company_short}-alerts.com",
                f"billing@{company_short}.com",
                f"support@{random.choice([company_short, fake.company().lower().replace(' ', '')])}.com",
                f"noreply@{company_short}-secure.com",
                f"{name.lower()}@{company_short}.com",
                f"admin@{company_short}.com"
            ]
            sender = random.choice(sender_options)
            
            # Select template
            fraud_category = random.choice(fraud_templates)
            subject_template = random.choice(fraud_category["subject_templates"])
            body_template = random.choice(fraud_category["body_templates"])
            
            # Determine if this email is actually legitimate (despite fraud template)
            is_fraud = random.random() < fraud_category["fraud_probability"]
            
            # Create parameters dictionary
            params = {
                "company": company,
                "company_name": company,
                "name": name,
                "last_name": last_name,
                "position": fake.job(),
                "project": fake.word().capitalize() + " " + fake.word(),
                "uuid": fake.uuid4(),
                "account_number": fake.credit_card_number(card_type=None),
                "routing_number": f"{fake.random_number(digits=9)}",
                "inv_num": invoice_number,
                "amount": fake.random_int(min=100, max=5000),
                "service_name": fake.catch_phrase(),
                "due_date": fake.date_this_year(),
                "customer_id":(fake.random_int(min=10000, max=99999)),
                "topic": fake.catch_phrase(),
                "month": fake.month_name(),
                "year": str(fake.random_int(min=2020, max=2023)),
                "follow_up_date": fake.date_this_year(),
                "next_steps": fake.sentence(),
                "event_name": fake.catch_phrase(),
                "event_date": fake.date_this_year(),
                "event_time": fake.time(),
                "event_location": fake.address(),
                "percentage": str(fake.random_int(min=30, max=70)),
                "meeting_date": fake.date_this_year(),
                "project_name": fake.word().capitalize() + " " + fake.word()
            }
            
            # Format subject and body
            subject = subject_template.format(**params)
            body = body_template.format(**params)
            
            # Add realistic elements that make it harder to detect
            if random.random() < 0.3:  # 30% have legitimate elements
                # Add random legitimate element to fraud email
                legitimate_elements = [
                    f"\n\nFor your reference, your account number is: {fake.credit_card_number(card_type=None)}",
                    f"\n\nThis communication is in accordance with our privacy policy: https://{company_short}.com/privacy",
                    f"\n\nQuestions? Visit our FAQ page at https://{company_short}.com/faq or call 1-800-{fake.random_number(digits=7)}"
                ]
                body += random.choice(legitimate_elements)
            
        else:  # legitimate
            # Create legitimate sender domains
            sender_options = [
                f"{name.lower()}.{last_name[:5].lower()}@{domain}",
                f"{random.choice(['info', 'support', 'contact', 'help', 'admin'])}@{domain}",
                f"{random.choice(['hr', 'billing', 'sales', 'marketing', 'service'])}@{domain}",
                f"{name.lower()}@{random.choice([domain, f'{company_short}.net'])}"
            ]
            sender = random.choice(sender_options)
            
            # Select template
            legit_category = random.choice(legitimate_templates)
            subject_template = random.choice(legit_category["subject_templates"])
            body_template = random.choice(legit_category["body_templates"])
            
            # Create parameters dictionary
            params = {
                "company": company,
                "company_name": company,
                "name": name,
                "last_name": last_name,
                "position": fake.job(),
                "uuid": fake.uuid4(),
                "project": fake.word().capitalize() + " " + fake.word(),
                "account_number": fake.credit_card_number(card_type=None),
                "routing_number": f"{fake.random_number(digits=9)}",
                "inv_num": invoice_number,
                "amount": fake.random_int(min=100, max=5000),
                "service_name": fake.catch_phrase(),
                "due_date": fake.date_this_year(),
                "customer_id": fake.random_int(min=10000, max=99999),
                "topic": fake.catch_phrase(),
                "month": fake.month_name(),
                "year": str(fake.random_int(min=2020, max=2023)),
                "follow_up_date": fake.date_this_year(),
                "next_steps": fake.sentence(),
                "event_name": fake.catch_phrase(),
                "event_date": fake.date_this_year(),
                "event_time": fake.time(),
                "event_location": fake.address(),
                "percentage": str(fake.random_int(min=30, max=70)),
                "meeting_date": fake.date_this_year(),
                "project_name": fake.word().capitalize() + " " + fake.word()
            }
            
            # Format subject and body
            subject = subject_template.format(**params)
            body = body_template.format(**params)
            
            is_fraud = False
        
        # Add signature to emails that typically have them
        if email_type == 'legitimate' or random.random() < 0.4:
            signature = f"""
Best regards,
{name} {last_name}
{fake.job()}
{company}
Phone: {fake.phone_number()}
            """.strip()
            body = f"{body}\n\n{signature}"
        
        # Calculate DMARC/SPF pass with realistic probability
        if email_type == 'fraud':
            # Fraudulent emails have lower pass rate but with variation
            dmarc_spf_pass = 1 if random.random() < 0.3 else 0  # 30% pass rate for fraud
        else:
            # Legitimate emails have higher pass rate but not always 100%
            dmarc_spf_pass = 1 if random.random() < 0.9 else 0  # 90% pass rate for legitimate
        
        # Extract features
        num_links = len(re.findall(r'https?://[^\s<>"\'()]+', body))
        num_dollar_signs = body.count('$')
        num_urgent_terms = len(re.findall(r'\b(urgent|immediately|asap|act now|verify|confirm|alert|required|suspended)\b', body, re.I))
        num_personal_pronouns = len(re.findall(r'\b(I|me|my|mine|we|us|our|ours)\b', body, re.I))
        
        # Check domain match (may not exist in all cases)
        domain_match = 0
        if '@' in sender:
            domain_part = sender.split('@')[1]
            domain_match = 1 if company_short in domain_part else 0
        
        # Check for suspicious subject patterns (include some legitimate ones)
        suspicious_subject = 0
        urgent_keywords = ['urgent', 'immediately', 'action required', 'security alert', 'account suspended']
        legitimate_keywords = ['reminder', 'update', 'invitation', 'confirmation', 'follow-up']
        
        if any(keyword in subject.lower() for keyword in urgent_keywords):
            suspicious_subject = 1
        elif any(keyword in subject.lower() for keyword in legitimate_keywords):
            suspicious_subject = -1  # Legitimate indicator
            
        # Check for signatures that don't match the sender
        signature_mismatch = 0
        if email_type == 'fraud' and random.random() < 0.4:  # 40% of fraud have mismatch
            signature_mismatch = 1
        elif "Best regards," in body and name.lower() not in body:
            signature_mismatch = 1
            
        # Create uncertainty for some attributes
        if email_type == 'fraud' and random.random() < 0.2:  # 20% of fraud emails
            domain_match = 1  # Sometimes matches
            suspicious_subject = -1  # Sometimes looks legitimate
            
        # Add feature vector
        emails.append([
            sender,
            subject,
            body,
            dmarc_spf_pass,
            int(is_fraud),
            num_links,
            num_dollar_signs,
            num_urgent_terms,
            num_personal_pronouns,
            domain_match,
            suspicious_subject,
            signature_mismatch
        ])
    
    return emails

def main():
    num_samples = 1000  # Adjust as needed
    print(f"Generating {num_samples} synthetic email samples with advanced variations...")
    
    # Generate emails
    email_data = generate_advanced_emails(num_samples)
    
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
    
    df = pd.DataFrame(email_data, columns=columns)
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save to CSV
    output_file = "advanced_synthetic_email_fraud_dataset.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Successfully generated advanced synthetic email dataset: '{output_file}'")
    print(f"Dataset contains {len(df)} samples with {df['label'].sum()} fraudulent emails")
    print(f"Percentage of fraudulent emails: {df['label'].mean()*100:.2f}%")
    
    # Display statistics
    print("\nDataset Statistics:")
    print(f"DMARC/SPF Pass Rate: {df['dmarc_spf_pass'].mean()*100:.2f}%")
    print(f"Average Number of Links: {df['num_links'].mean():.2f}")
    print(f"Average Number of Dollar Signs: {df['num_dollar_signs'].mean():.2f}")
    print(f"Average Number of Urgent Terms: {df['num_urgent_terms'].mean():.2f}")
    print(f"Average Number of Personal Pronouns: {df['num_personal_pronouns'].mean():.2f}")
    print(f"Domain Match Rate: {df['domain_match'].mean()*100:.2f}%")
    print(f"Suspicious Subject Rate: {df[df['suspicious_subject'] == 1].shape[0]}/{len(df)} ({df[df['suspicious_subject'] == 1].shape[0]/len(df)*100:.2f}%)")
    print(f"Signature Mismatch Rate: {df['signature_mismatch'].mean()*100:.2f}%")

if __name__ == "__main__":
    main()