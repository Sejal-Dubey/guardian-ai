import sys
import os
import torch

# --- Local imports ---
from audio_intent_detection import transcribe_audio, classify_intent, HeuristicEngine
from antispoofing_detection import analyze_single_file

# Paths
CONFIG_FILE = "aasist/config/AASIST-L.conf"
WEIGHTS_FILE = "aasist/models/weights/AASIST-L.pth"

def run_pipeline(audio_file: str):
    print("\n=== Guardian-AI Voice Pipeline ===")

    # Step 1: Transcribe with Whisper
    transcript = transcribe_audio(audio_file)
    print(f"[TRANSCRIPT]\n{transcript}\n")

    
    # Step 4: Anti-spoof detection (AASIST)
    spoof_result = analyze_single_file(audio_file, CONFIG_FILE, WEIGHTS_FILE)
    antispoof_score = spoof_result["spoof"] if spoof_result else 0.0
    print(f"[ANTI-SPOOF] Spoof Score: {antispoof_score:.4f}")

    # Step 2: Intent detection (RoBERTa)
    roberta_result = classify_intent(transcript)
    intent_score = roberta_result["suspicious_score"]
    print(f"[INTENT DETECTION] Suspicious Score: {intent_score:.4f}")

    # Step 3: Heuristic scoring
    heuristic = HeuristicEngine()
    heuristic_score = heuristic.score(transcript)
    print(f"[HEURISTIC ENGINE] Suspicious Score: {heuristic_score:.4f}")


    # Step 5: Weighted fusion
    final_score = (
        0.50  * antispoof_score +
        0.35 * intent_score +
        0.15 * heuristic_score
    )

    decision = "Suspicious" if final_score > 0.5 else "Benign"

    print("\n=== Final Decision ===")
    print(f"Weighted Suspicious Score: {final_score:.4f}")
    print(f"Decision: {decision}")

    return {
        "transcript": transcript,
        "antispoof_score": antispoof_score,
        "intent_score": intent_score,
        "heuristic_score": heuristic_score,
        "final_score": final_score,
        "decision": decision
    }


if __name__ == "__main__":
    audio_file = "test_audio0.mp3"   
    run_pipeline(audio_file)
