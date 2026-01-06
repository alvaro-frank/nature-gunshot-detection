import os
import numpy as np
import librosa
from tqdm import tqdm  # progress bar

# --- CONFIGURATION ---
SAMPLE_RATE = 22050
DURATION = 2.0  # We cut/pad all audio to exactly 2 seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
INPUT_DIR = "augmented_data"
OUTPUT_DIR = "processed_data"

def preprocess_audio(file_path):
    try:
        # 1. Load Audio
        # librosa.load automatically resamples to sr=22050
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # 2. Fix Length (Pad or Truncate)
        if len(audio) < SAMPLES_PER_TRACK:
            # Pad with zeros if too short
            padding = int(SAMPLES_PER_TRACK - len(audio))
            audio = np.pad(audio, (0, padding))
        else:
            # Truncate if too long
            audio = audio[:int(SAMPLES_PER_TRACK)]
            
        # 3. Convert to Mel-Spectrogram
        # n_mels=128 is standard for these models
        mels = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        
        # 4. Convert to Decibels (Log scale) - Critical for human-like hearing
        mels_db = librosa.power_to_db(mels, ref=np.max)
        
        # Normalization (Optional but good: keep values between 0 and 1 roughly)
        # For now, we keep raw dB values (-80 to 0 usually)
        return mels_db
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    # Create output directories
    categories = ["gunshots", "nature_noise"]
    
    for category in categories:
        # Setup paths
        input_folder = os.path.join(INPUT_DIR, category)
        output_folder = os.path.join(OUTPUT_DIR, category)
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"Processing category: {category}...")
        
        # Get list of files
        files = [f for f in os.listdir(input_folder) if f.endswith('.wav') or f.endswith('.mp3')]
        
        if len(files) == 0:
            print(f"⚠️ WARNING: No files found in {input_folder}. Please add files!")
            continue

        # Process each file
        for f in tqdm(files):
            file_path = os.path.join(input_folder, f)
            
            # Generate spectrogram
            spectrogram = preprocess_audio(file_path)
            
            if spectrogram is not None:
                # Save as .npy (Numpy Array)
                # We save with the same filename but .npy extension
                save_name = f.replace('.wav', '.npy').replace('.mp3', '.npy')
                np.save(os.path.join(output_folder, save_name), spectrogram)

if __name__ == "__main__":
    main()