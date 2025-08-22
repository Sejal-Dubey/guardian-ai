import torch
import torchaudio
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from aasist.models.AASIST import Model


def preprocess_audio(audio_path, target_sr=16000, force_mono=True):
    """
    Load, resample, and (optionally) fold to mono.
    Returns a dict with two tensors ready for model input:
      - batch_T: shape [B, T]
      - batch_C_T: shape [B, 1, T]
    """

    waveform, sr = torchaudio.load(audio_path)  # [C, T] or [T]
    print(f"[DEBUG] loaded waveform shape={tuple(waveform.shape)}, sr={sr}")

    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
        sr = target_sr
        print(f"[DEBUG] resampled waveform shape={tuple(waveform.shape)}, sr={sr}")

    if force_mono and waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]
        print(f"[DEBUG] folded-to-mono shape={tuple(waveform.shape)}")

    waveform = waveform.to(torch.float32).contiguous()

    if waveform.dim() == 1:
        # [T]
        batch_T = waveform.unsqueeze(0)            # [1, T]
        batch_C_T = waveform.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
    elif waveform.dim() == 2:
        # [C, T] (C should be 1 after mono fold)
        if waveform.size(0) != 1:
            # If still multi-channel, force mono now
            waveform = waveform.mean(dim=0, keepdim=True)
            print(f"[DEBUG] forced-mono (late) shape={tuple(waveform.shape)}")

        batch_T = waveform.squeeze(0).unsqueeze(0)      # [1, T]
        batch_C_T = waveform.unsqueeze(0)               # [1, 1, T]
    else:
        raise ValueError(f"Unexpected waveform shape: {waveform.shape}")

    print(f"[DEBUG] candidate shapes: batch_T={tuple(batch_T.shape)}, batch_C_T={tuple(batch_C_T.shape)}")
    return {"batch_T": batch_T, "batch_C_T": batch_C_T}


def forward_with_fallback(model, candidates):
    """
    Try forwarding with [B, T] first (2-D), then fallback to [B, 1, T] (3-D).
    Returns logits (tensor) and which path worked.
    """
    model.eval()
    with torch.no_grad():
        try:
            print("[DEBUG] trying input shape [B, T] ->", tuple(candidates["batch_T"].shape))
            out = model(candidates["batch_T"])
            which = "[B, T]"
            return out, which
        except Exception as e1:
            print(f"[DEBUG] [B, T] failed: {e1}")

        print("[DEBUG] trying input shape [B, 1, T] ->", tuple(candidates["batch_C_T"].shape))
        out = model(candidates["batch_C_T"])
        which = "[B, 1, T]"
        return out, which


def analyze_single_file(audio_path, config_path, model_weights_path):
    """
    Load AASIST-L, run on a single file, and print bonafide/spoof probabilities.
    """
    try:
        with open(config_path, "r") as f_yaml:
            config = yaml.safe_load(f_yaml)


        model = Model(config["model_config"])
        print(f"Loading pre-trained weights from: {model_weights_path}")
        state = torch.load(model_weights_path, map_location="cpu")
        model.load_state_dict(state)
        print("Model loaded successfully.")

        print(f"Loading audio file: {audio_path}")
        candidates = preprocess_audio(audio_path, target_sr=16000, force_mono=True)

        outputs, used_shape = forward_with_fallback(model, candidates)
        print(f"[DEBUG] model forward succeeded with input {used_shape}")

        logits = outputs[0] if isinstance(outputs, tuple) else outputs  
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)  

        probs = torch.softmax(logits, dim=1).squeeze(0).tolist()  # [2]
        bonafide, spoof = probs[0], probs[1]

        print("\n--- Analysis Complete ---")
        print(f"File: {audio_path}")
        print(f"Input shape used: {used_shape}")
        print(f"Bonafide Score: {bonafide:.4f}")
        print(f"Spoof Score:    {spoof:.4f}")

        return {"file": audio_path, "bonafide": bonafide, "spoof": spoof}

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    config_file = "aasist/config/AASIST-L.conf"
    weights_file = "aasist/models/weights/AASIST-L.pth"
    file_to_test = "test_audio1.mp3"  

    analyze_single_file(file_to_test, config_file, weights_file)
