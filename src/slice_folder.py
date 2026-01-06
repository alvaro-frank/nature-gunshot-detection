import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
# 1. Put your long recordings (50min file, etc.) in this folder:
INPUT_FOLDER = "raw_data/imports" 

# 2. The script will dump the 2-second chunks here:
OUTPUT_FOLDER = "raw_data/nature_noise" 

CHUNK_DURATION = 2.0  # Seconds
SR = 22050           # Sample Rate
MIN_VOLUME = 0.005   # Skip chunks that are pure silence

def process_folder():
    # Create folders if they don't exist
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"⚠️ Created folder '{INPUT_FOLDER}'. Please put your audio files there and run this again.")
        return
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get list of all audio files
    supported_extensions = ('.wav', '.mp3', '.flac', '.ogg')
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(supported_extensions)]
    
    if not files:
        print(f"❌ No audio files found in '{INPUT_FOLDER}'")
        return

    print(f"📂 Found {len(files)} files to slice.")
    
    # Iterate through every file in the imports folder
    for filename in files:
        file_path = os.path.join(INPUT_FOLDER, filename)
        file_base_name = os.path.splitext(filename)[0]
        
        print(f"   ✂️ Slicing: {filename}...")
        
        try:
            # Load the full audio file
            audio, _ = librosa.load(file_path, sr=SR)
        except Exception as e:
            print(f"     ⚠️ Error loading {filename}: {e}")
            continue

        total_samples = len(audio)
        chunk_samples = int(CHUNK_DURATION * SR)
        num_chunks = total_samples // chunk_samples
        
        saved_count = 0
        
        # Slice it up
        for i in range(num_chunks):
            start = i * chunk_samples
            end = start + chunk_samples
            chunk = audio[start:end]
            
            # Skip silence (optional volume check)
            if np.max(np.abs(chunk)) < MIN_VOLUME:
                continue
            
            # Save: naming format is "originalfilename_slice_001.wav"
            out_name = f"{file_base_name}_slice_{i:04d}.wav"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)
            sf.write(out_path, chunk, SR)
            saved_count += 1
            
        print(f"     ✅ Created {saved_count} clips from {filename}")

    print("\n🎉 All files processed! Run 'augment.py' next.")

if __name__ == "__main__":
    process_folder()