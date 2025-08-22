import re
import torch
import torchaudio
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# --- Load your fine-tuned RoBERTa ---
roberta_model_path = "mufg_roberta_intent"
tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path)
roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_path)

# --- Load Whisper for transcription ---
whisper_model_name = "openai/whisper-small"  # you can use "large" for better accuracy
processor = WhisperProcessor.from_pretrained(whisper_model_name)
whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)

# --- Heuristic Engine ---
class HeuristicEngine:
    def __init__(self):
        # Suspicious patterns/keywords
        self.suspicious_keywords = [
            "otp", "password", "pin", "account", "verify", "urgent", 
            "immediate", "action required", "suspicious activity",
            "transaction", "block", "security", "bank", "account freeze", "update",
            "fraud", "restricted", "suspend", "id", "login"
        ]
        self.otp_pattern = re.compile(r"\b\d{4,8}\b")

    def score(self, transcript: str) -> float:
        text = transcript.lower()
        score = 0.0


        keyword_hits = [kw for kw in self.suspicious_keywords if kw in text]
        score += 0.1 * len(keyword_hits)

        if self.otp_pattern.search(text):
            score += 0.3

        return min(1.0, score)


def transcribe_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    predicted_ids = whisper_model.generate(inputs["input_features"])
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def classify_intent(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = roberta_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze()

    benign_score, suspicious_score = probs.tolist()
    return {
        "text": text,
        "benign_score": round(benign_score, 4),
        "suspicious_score": round(suspicious_score, 4),
        "prediction": "Suspicious" if suspicious_score > benign_score else "Benign"
    }



# --- Run Pipeline ---
if __name__ == "__main__":
    audio_file = "test_audio0.mp3"

    # Transcribe with Whisper
    transcript = transcribe_audio(audio_file)

    # Intent detection
    roberta_result = classify_intent(transcript)

    # Heuristic scoring
    heuristic = HeuristicEngine()
    heuristic_score = heuristic.score(transcript)

    # Step 4: Fuse results (simple weighted average, can be tuned)
    final_score = (roberta_result["suspicious_score"] + heuristic_score) / 2

    print("\n--- Audio Intent Analysis ---")
    print(f"Transcript: {transcript}")
    print(f"RoBERTa → Prediction: {roberta_result['prediction']} | Suspicious Score: {roberta_result['suspicious_score']}")
    print(f"Heuristic → Suspicious Score: {heuristic_score}")
    print(f"Final Combined Suspicious Score: {round(final_score, 4)}")
    print("Final Decision:", "Suspicious" if final_score > 0.5 else "Benign")