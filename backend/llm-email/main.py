import socketio
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# ===================================================================
# 1. LOAD YOUR TRAINED MODELS AND FILES ONCE AT STARTUP
# ===================================================================
print("INFO: Server is starting up...")

# HuggingFace model + tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
hf_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# Example: scikit-learn model
try:
    clf = joblib.load("model.pkl")
    print("INFO: model.pkl loaded successfully")
except Exception as e:
    clf = None
    print("WARNING: Could not load model.pkl →", e)


# ===================================================================
# 2. FASTAPI APP
# ===================================================================
app = FastAPI(title="Guardian AI Backend", version="1.0")


@app.get("/")
async def root():
    return {"message": "Backend is running!"}


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(req: PredictRequest):
    """
    Example POST endpoint for predictions
    """
    inputs = tokenizer(req.text, return_tensors="pt")
    with torch.no_grad():
        emb = hf_model(**inputs).last_hidden_state.mean(dim=1).numpy()

    if clf:
        pred = clf.predict(emb)[0]
        return {"prediction": int(pred), "embedding": emb.tolist()}
    else:
        return {"error": "Model not loaded", "embedding": emb.tolist()}


# ===================================================================
# 3. SOCKET.IO SERVER (ASYNC MODE)
# ===================================================================
sio = socketio.AsyncServer(cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio)


@sio.event
async def connect(sid, environ):
    print(f"✅ Socket.IO client connected: {sid}")


@sio.event
async def disconnect(sid):
    print(f"❌ Socket.IO client disconnected: {sid}")


@sio.event
async def message(sid, data):
    print(f"📩 Message from {sid}: {data}")
    await sio.emit("response", {"echo": data}, to=sid)


# ===================================================================
# 4. MOUNT SOCKET.IO INTO FASTAPI
# ===================================================================
# 👉 Do not overwrite app, mount under `/ws`
app.mount("/ws", socket_app)
