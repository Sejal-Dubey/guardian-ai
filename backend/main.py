# backend/main.py
import socketio
from fastapi import FastAPI

# Create FastAPI app
app = FastAPI()

# Create Socket.IO asynchronous server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins='*')
socket_app = socketio.ASGIApp(sio)

@app.get("/")
def read_root():
    return {"message": "Guardian AI Backend is running"}

# Mount the Socket.IO app
app.mount("/", socket_app)

@sio.event
async def connect(sid, environ):
    print(f"Socket.IO client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Socket.IO client disconnected: {sid}")