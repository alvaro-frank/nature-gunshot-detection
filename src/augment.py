import os
import random
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

# --- CONFIGURATION ---
CLEAN_GUN_DIR = "raw_data/gunshots"
BACKGROUND_DIR = "raw_data/nature_noise"
OUTPUT_DIR = "augmented_data"
SAMPLE_RATE = 22050
TARGET_DURATION = 2.0 # Seconds
NUM_AUGMENTATIONS = 5 # How many "remixes" to make per gunshot file

def load_random_background(duration_samples):
    """Picks a random nature file and cuts a random snippet from it."""
    bg_files = [f for f in os.listdir(BACKGROUND_DIR) if f.endswith('.wav')]
    if not bg_files:
        return np.zeros(duration_samples)
    
    bg_file = random.choice(bg_files)
    path = os.path.join(BACKGROUND_DIR, bg_file)
    
    # Load fast (just getting duration first would be better, but this is simple)
    bg, _ = librosa.load(path, sr=SAMPLE_RATE)
    
    # If background is too short, loop it
    if len(bg) < duration_samples:
        tile_factor = int(np.ceil(duration_samples / len(bg)))
        bg = np.tile(bg, tile_factor)
        
    # Crop a random section
    start = random.randint(0, len(bg) - duration_samples)
    return bg[start : start + duration_samples]

def mix_audio(foreground, background, snr_db):
    """Mixes signal with noise at a specific Signal-to-Noise Ratio (SNR)."""
    # Calculate power
    fore_power = np.mean(foreground ** 2)
    back_power = np.mean(background ** 2)
    
    if back_power == 0: return foreground

    # Calculate required noise scalar
    target_ratio = 10 ** (snr_db / 10)
    scalar = np.sqrt(fore_power / (target_ratio * back_power))
    
    return foreground + (background * scalar)

def main():
    # Setup Output Folders
    os.makedirs(os.path.join(OUTPUT_DIR, "gunshots"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "nature_noise"), exist_ok=True) # We just copy pure nature here
    
    gun_files = [f for f in os.listdir(CLEAN_GUN_DIR) if f.endswith('.wav')]
    
    print(f"--- STARTING AUGMENTATION ---")
    print(f"Found {len(gun_files)} clean gunshots.")
    print(f"Generating {len(gun_files) * NUM_AUGMENTATIONS} synthetic training files...")

    for f in tqdm(gun_files):
        path = os.path.join(CLEAN_GUN_DIR, f)
        gun, _ = librosa.load(path, sr=SAMPLE_RATE)
        
        # Normalize Gunshot length (pad/cut to 2s)
        target_len = int(SAMPLE_RATE * TARGET_DURATION)
        if len(gun) > target_len:
            gun = gun[:target_len]
        else:
            gun = np.pad(gun, (0, target_len - len(gun)))

        # Create 5 variations of this ONE gun
        for i in range(NUM_AUGMENTATIONS):
            # 1. Get Background
            noise = load_random_background(target_len)
            
            # 2. Random SNR (Signal to Noise Ratio)
            # 10dB = Gun is 2x louder than rain (Easy)
            # -5dB = Gun is quieter than rain (Hard/Distant)
            snr = random.uniform(-5, 15) 
            
            # 3. Mix
            mixed = mix_audio(gun, noise, snr)
            
            # 4. Save
            out_name = f"{f[:-4]}_aug_{i}.wav"
            out_path = os.path.join(OUTPUT_DIR, "gunshots", out_name)
            sf.write(out_path, mixed, SAMPLE_RATE)

    print("\n✅ Augmentation Complete!")
    print("Now run 'preprocess.py' but point it to the 'augmented_data' folder!")

if __name__ == "__main__":
    main()