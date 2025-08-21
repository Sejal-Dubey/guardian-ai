from datasets import load_dataset
import soundfile as sf
import os

def download_and_save_first_sample(dataset_name, split="train"):
    """
    Downloads the first audio sample from a dataset and saves it as a .wav file.
    """
    try:
        print(f"INFO: Loading the first sample from '{dataset_name}'...")
        # Use split="train[:1]" to download only the first sample
        single_sample_dataset = load_dataset(dataset_name, split=f"{split}[:1]")
        
        # Access the audio data from the first sample
        sample = single_sample_dataset[0]
        audio_data = sample['audio']['array']
        sample_rate = sample['audio']['sampling_rate']
        
        # Define the output filename
        output_filename = f"{dataset_name.replace('/', '_')}_sample_0.wav"
        
        # Save the audio data as a .wav file
        sf.write(output_filename, audio_data, sample_rate)
        
        print(f"\n--- ✅ Success! ---")
        print(f"Audio sample saved as: {os.path.abspath(output_filename)}")

    except Exception as e:
        print(f"\n--- ❌ ERROR ---")
        print(f"An error occurred: {e}")

# --- Main execution block ---
if __name__ == "__main__":
    # The name of the dataset from Hugging Face
    dataset_to_load = "mueller91/MLAAD"
    
    download_and_save_first_sample(dataset_to_load)