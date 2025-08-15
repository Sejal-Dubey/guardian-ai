# backend/ai_pipelines.py
import time

def analyze_text(email_content: str):
    """
    This is a DUMMY function that simulates the text analysis pipeline.
    It will be replaced by Sejal's real code later.
    """
    print("(Placeholder) AI Pipeline: Analyzing text...")
    time.sleep(1) # Simulate a short delay
    return {"risk_score": 0.88, "evidence": "Detected urgent language."}

def analyze_voice(audio_file_path: str):
    """
    This is a DUMMY function that simulates the voice analysis pipeline.
    It will be replaced by Sejal's real code later.
    """
    print(f"(Placeholder) AI Pipeline: Analyzing voice file at {audio_file_path}...")
    time.sleep(1) # Simulate a short delay
    return {"risk_score": 0.78, "evidence": "Detected audio synthesis."}