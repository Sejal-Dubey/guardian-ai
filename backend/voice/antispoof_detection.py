import torch
import torchaudio
import yaml
import sys
import os

# This adds your backend folder to Python's path to help find the 'model' folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the correct, complex model architecture from the local 'model' folder
from aasist.models.AASIST import Model 

def analyze_single_file(audio_path, config_path, model_weights_path):
    """
    Loads the CORRECT pre-trained AASIST-L model and analyzes a single audio file.
    """
    try:
        # 1. Load the configuration file
        with open(config_path, "r") as f_yaml:
            config = yaml.safe_load(f_yaml)

        # 2. Load the model architecture using the config
        model = Model(config["model_config"])
        
        # 3. Load the pre-trained weights from the LOCAL FILE
        print(f"Loading pre-trained weights from: {model_weights_path}")
        model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
        model.eval()
        print("Model loaded successfully.")

        # 4. Load and prepare the audio file
        print(f"Loading audio file: {audio_path}")
        signal, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            signal = resampler(signal)

        # 5. Get the model's prediction
        print("Analyzing audio...")

        with torch.no_grad():
            # signal shape: [channels, samples]
            if signal.ndim == 2:  # e.g., [1, T]
                signal = signal.squeeze(0)   # → [T]
            # ensure batch dimension
            signal = signal.unsqueeze(0)     # → [1, T]

            print(f"Input to model: {signal.shape}")  # should be [1, T]
            output_scores = model(signal)


        # 6. Convert output to a spoof probability score
        output_scores = model(signal)

        # If model returns (logits, features), take the first
        if isinstance(output_scores, tuple):
            logits = output_scores[0]
        else:
            logits = output_scores

        probabilities = torch.nn.functional.softmax(logits, dim=1)
        spoof_score = probabilities[0][1].item() # 0 is bonafide, 1 is spoof



        print(f"\n--- Analysis Complete ---")
        print(f"File: {audio_path}")
        print(f"Predicted Spoof Score: {spoof_score:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    # Path to the config file from the AASIST repo
    config_file = "aasist/config/AASIST-L.conf"

    # Path to the weights file you downloaded
    weights_file = r"aasist/models/weights/AASIST-L.pth"

    # Your audio file to test
    file_to_test = r"ElevenLabs_Text_to_Speech_audio.wav"
    
    # Run the analysis
    analyze_single_file(file_to_test, config_file, weights_file)