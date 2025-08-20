import socketio
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ===================================================================
# 1. LOAD YOUR TRAINED MODELS AND FILES ONCE AT STARTUP
# ===================================================================
print("INFO: Server is starting up...")

# Load the saved model, which now contains the model and the threshold
try:
    model_info = joblib.load(r'C:\Users\Sejal\Downloads\guardian-ai\guardian-ai\backend\email\fraud_detection_model_advanced.joblib')
    email_model = model_info['model']
    prediction_threshold = model_info['threshold']
    print("INFO: Advanced Random Forest/Gradient Boosting model loaded successfully.")
    print(f"INFO: Using optimal prediction threshold: {prediction_threshold}")
except FileNotFoundError:
    print("ERROR: 'fraud_detection_model_advanced.joblib' not found.")
    email_model = None

# Load the saved style fingerprint
try:
    style_fingerprint = np.load(r'C:\Users\Sejal\Downloads\guardian-ai\guardian-ai\backend\email\style_fingerprint_advanced.npy')
    print("INFO: Advanced style fingerprint loaded successfully.")
except FileNotFoundError:
    print("ERROR: 'style_fingerprint_advanced.npy' not found.")
    style_fingerprint = None

# Load the RoBERTa model for embeddings
print("INFO: Loading RoBERTa model...")
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
roberta_model = AutoModel.from_pretrained('roberta-base')
print("INFO: RoBERTa model loaded.")
print("--- Server startup complete. Ready for requests. ---")


# ===================================================================
# 2. DEFINE THE AI PIPELINE FUNCTIONS (MATCHING TRAINING SCRIPT)
# ===================================================================

def get_roberta_embedding(subject, body):
    combined_text = f"Subject: {subject}\n\n{body}"
    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=128, padding='max_length')
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    return outputs.last_hidden_state[0][0].numpy()

def get_header_score(sender_email, domain_match):
    header_score = 0.0
    try:
        domain = sender_email.split('@')[1]
        suspicious_keywords = ['-security', '-support', '-alerts', '-verify', 'secure-', 'account', 'login', 'update']
        if any(keyword in domain for keyword in suspicious_keywords):
            header_score += 0.3
        legitimate_domains = ['microsoft.com', 'apple.com', 'amazon.com', 'paypal.com', 'chase.com', 'bankofamerica.com']
        if any(legit_domain in domain for legit_domain in legitimate_domains) and domain_match == 0:
            header_score += 0.4
    except IndexError:
        header_score += 0.5
    return min(header_score, 1.0)

def get_authorship_score(email_embedding, style_fingerprint):
    if style_fingerprint is None: return 0.5
    similarity = cosine_similarity(email_embedding.reshape(1, -1), style_fingerprint.reshape(1, -1))[0][0]
    return 1.0 - similarity

def analyze_email_real(sender, subject, body):
    """
    Analyzes an email by recreating the full feature vector from the training script
    and using the loaded model to predict the fraud score.
    """
    if not email_model or style_fingerprint is None:
        return {"risk_score": -1, "evidence": "Model or fingerprint not loaded."}
        
    print("EMAIL_PIPELINE: Starting real analysis...")
    
    # 1. Generate core features
    embedding_vector = get_roberta_embedding(subject, body)
    
    # For a live request, we have to simulate the other features
    domain_match = 0 # Assume mismatch for a new sender
    header_score = get_header_score(sender, domain_match)
    authorship_score = get_authorship_score(embedding_vector, style_fingerprint)

    # 2. Simulate additional features based on the text
    # In a real app, these would be calculated more robustly
    num_links = 1 if 'http' in body else 0
    num_dollar_signs = body.count('$')
    num_urgent_terms = sum(body.lower().count(term) for term in ['urgent', 'immediate', 'action required'])
    num_personal_pronouns = sum(body.lower().count(pronoun) for pronoun in [' i ', ' my ', ' me '])
    suspicious_subject = 1 if any(term in subject.lower() for term in ['alert', 'security', 'verify']) else 0
    signature_mismatch = 1 if 'thanks' not in body.lower() else 0

    additional_features = [
        num_links, num_dollar_signs, num_urgent_terms, num_personal_pronouns,
        domain_match, suspicious_subject, signature_mismatch
    ]
    
    # 3. Combine into the final feature vector
    feature_vector = np.concatenate([embedding_vector, [header_score], [authorship_score], additional_features]).reshape(1, -1)
    
    # 4. Get probability score from the trained model
    probability_scores = email_model.predict_proba(feature_vector)
    email_risk_score = probability_scores[0][1] # Probability of fraud
    
    print(f"EMAIL_PIPELIPNE: Calculated risk score: {email_risk_score:.4f}")
    return {"risk_score": email_risk_score, "evidence": "Analysis complete."}

def analyze_voice_hardcoded():
    return {"risk_score": 0.85, "evidence": "Hardcoded result: Signs of audio synthesis."}

# ===================================================================
# 3. FASTAPI SERVER LOGIC
# ===================================================================

app = FastAPI()
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)

class AnalysisRequest(BaseModel):
    sender: str
    subject: str
    body: str

@app.post("/analyze")
async def analyze_data(request: AnalysisRequest):
    print(f"\n--- Received API request ---")
    await sio.start_background_task(run_full_analysis, request.sender, request.subject, request.body)
    return {"message": "Analysis started. Check terminal for output."}

async def run_full_analysis(sender, subject, body):
    print("--- FUSION_ENGINE: Starting analysis ---")
    
    voice_result = analyze_voice_hardcoded()
    email_result = analyze_email_real(sender, subject, body)
    
    voice_score = voice_result['risk_score']
    email_score = email_result['risk_score']
    base_score = max(voice_score, email_score)
    final_score = base_score
    
    if voice_score > 0.7 and email_score > 0.7:
        final_score = min(1.0, base_score + 0.20)
        print("FUSION_ENGINE: Boost applied.")
    
    print(f"--- FUSION_ENGINE: Analysis Complete ---")
    print(f"  - Voice Score: {voice_score:.4f} (Hardcoded)")
    print(f"  - Email Score: {email_score:.4f} (Real Model)")
    print(f"  - FINAL FUSED SCORE: {final_score:.4f}")

app.mount("/", socket_app)