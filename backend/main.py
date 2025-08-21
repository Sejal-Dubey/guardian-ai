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

# Use relative paths so the code works for everyone
MODEL_PATH = r'email/fraud_detection_model_advanced.joblib'
FINGERPRINT_PATH = r'email/style_fingerprint_advanced.npy'

try:
    model_info = joblib.load(MODEL_PATH)
    email_model = model_info['model']
    prediction_threshold = model_info['threshold']
    print("INFO: Advanced model loaded successfully.")
    print(f"INFO: Using optimal prediction threshold: {prediction_threshold}")
except FileNotFoundError:
    print(f"ERROR: Model not found at '{MODEL_PATH}'.")
    email_model = None

try:
    style_fingerprint = np.load(FINGERPRINT_PATH)
    print("INFO: Style fingerprint loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Fingerprint not found at '{FINGERPRINT_PATH}'.")
    style_fingerprint = None

print("INFO: Loading RoBERTa model...")
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
roberta_model = AutoModel.from_pretrained('roberta-base')
print("INFO: RoBERTa model loaded.")
print("--- Server startup complete. ---")


# ===================================================================
# 2. DEFINE THE AI PIPELINE FUNCTIONS
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
        if any(keyword in domain for keyword in suspicious_keywords): header_score += 0.3
        legitimate_domains = ['microsoft.com', 'apple.com', 'amazon.com', 'paypal.com']
        if any(legit_domain in domain for legit_domain in legitimate_domains) and domain_match == 0: header_score += 0.4
    except IndexError:
        header_score += 0.5
    return min(header_score, 1.0)

def get_authorship_score(email_embedding, style_fingerprint):
    if style_fingerprint is None: return 0.5
    similarity = cosine_similarity(email_embedding.reshape(1, -1), style_fingerprint.reshape(1, -1))[0][0]
    return 1.0 - similarity

def analyze_email_real(sender, subject, body):
    if not email_model or style_fingerprint is None:
        return {"risk_score": -1, "evidence": "Model or fingerprint not loaded."}
        
    embedding_vector = get_roberta_embedding(subject, body)
    domain_match = 0
    header_score = get_header_score(sender, domain_match)
    authorship_score = get_authorship_score(embedding_vector, style_fingerprint)
    
    # Simulate other features for live inference
    additional_features = [
        1 if 'http' in body else 0, body.count('$'),
        sum(body.lower().count(term) for term in ['urgent', 'immediate', 'action required']),
        sum(body.lower().count(pronoun) for pronoun in [' i ', ' my ', ' me ']),
        domain_match, 1 if any(term in subject.lower() for term in ['alert', 'security', 'verify']) else 0,
        1 if 'thanks' not in body.lower() else 0
    ]
    
    feature_vector = np.concatenate([embedding_vector, [header_score], [authorship_score], additional_features]).reshape(1, -1)
    
    probability_scores = email_model.predict_proba(feature_vector)
    email_risk_score = probability_scores[0][1]
    
    return {
        "risk_score": email_risk_score, 
        "evidence": {
            "Header Score": f"{header_score:.2f}",
            "Authorship Score": f"{authorship_score:.2f}",
            "Detected Intent": "Urgent Payment Request" if email_risk_score > 0.6 else "General Inquiry"
        }
    }

def analyze_voice_hardcoded():
    return {
        "risk_score": 0.85, 
        "evidence": {
            "Deepfake Analysis": "Signs of audio synthesis detected (Hardcoded).",
            "Scam Intent": "Urgent language detected (Hardcoded)."
        }
    }

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
    return {"message": "Analysis started."}

async def run_full_analysis(sender, subject, body):
    print("--- FUSION_ENGINE: Starting analysis ---")
    voice_result = analyze_voice_hardcoded()
    email_result = analyze_email_real(sender, subject, body)
    
    voice_score = voice_result['risk_score']
    email_score = email_result['risk_score']
    base_score = max(voice_score, email_score)
    final_score = base_score
    boost_applied = False
    
    if voice_score > 0.7 and email_score > 0.7:
        final_score = min(1.0, base_score + 0.20)
        boost_applied = True
        print("FUSION_ENGINE: Boost applied.")
    
    result_payload = {
        "voiceResult": voice_result,
        "emailResult": email_result,
        "boostApplied": boost_applied,
        "finalScore": final_score
    }
    
    # --- THIS IS THE CRUCIAL LINE ---
    await sio.emit('analysis_result', result_payload)
    print("--- FUSION_ENGINE: Analysis Complete. Results pushed to clients. ---")

@sio.event
async def connect(sid, environ):
    print(f"Socket.IO client connected: {sid}")

app.mount("/", socket_app)