import numpy as np
import librosa
import tensorflow as tf
import os
import argparse
import keras

# --- CONFIGURATION ---
MODEL_PATH = "models/forest_gunshot_detector.keras"
SAMPLE_RATE = 22050
DURATION = 2.0        # Window size (matches training)
HOP_LENGTH = 0.5      # Step size (0.5s means we check overlapping windows)
THRESHOLD = 0.85      # Confidence threshold

def preprocess_segment(audio_segment):
    """
    Takes a 2-second audio array and converts it to a Spectrogram
    """
    # 1. Convert to Mel-Spectrogram
    mels = librosa.feature.melspectrogram(y=audio_segment, sr=SAMPLE_RATE, n_mels=128)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    
    # 2. Fix Dimensions (Pad or Crop to 87 width)
    target_width = 87
    if mels_db.shape[1] < target_width:
        mels_db = np.pad(mels_db, ((0,0), (0, target_width - mels_db.shape[1])))
    else:
        mels_db = mels_db[:, :target_width]
        
    # 3. Reshape for CNN input (1, 128, 87, 1)
    return mels_db.reshape(1, 128, target_width, 1)

def analyze_audio_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    print(f"Loading model: {MODEL_PATH}...")
    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Processing audio: {file_path}...")
    # Load the full audio file
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    duration_secs = len(audio) / sr
    print(f"Audio Duration: {duration_secs:.2f} seconds")

    # --- SLIDING WINDOW LOOP ---
    # We step through the file in 0.5s increments
    step_samples = int(HOP_LENGTH * SAMPLE_RATE)
    window_samples = int(DURATION * SAMPLE_RATE)
    
    detections = []
    
    # Iterate through the audio
    for i in range(0, len(audio) - window_samples, step_samples):
        # Extract 2-second chunk
        chunk = audio[i : i + window_samples]
        
        # --- VOLUME GATE (Optional but recommended) ---
        # Skip quiet segments to reduce false positives
        vol = np.sqrt(np.mean(chunk**2))
        if vol < 0.02: 
            continue

        # Prepare and Predict
        input_tensor = preprocess_segment(chunk)
        prediction = model.predict(input_tensor, verbose=0)[0][0]
        
        # Calculate timestamp
        current_time = i / SAMPLE_RATE
        
        if prediction > THRESHOLD:
            print(f"  [TIME {current_time:.1f}s] - 🔴 GUNSHOT DETECTED (Conf: {prediction*100:.1f}%)")
            detections.append((current_time, prediction))
        else:
            # Optional: Print clean segments to debug
            # print(f"  [TIME {current_time:.1f}s] - ... Clean (Conf: {prediction*100:.1f}%)")
            pass

    # --- FINAL SUMMARY ---
    print("\n--- ANALYSIS REPORT ---")
    if len(detections) > 0:
        print(f"⚠️  ALERT: {len(detections)} potential gunshots detected.")
        print(f"First detection at: {detections[0][0]:.1f} seconds.")
    else:
        print("✅ CLEAN: No gunshots detected in this file.")

if __name__ == "__main__":
    # You can hardcode your file here if you don't want to use terminal arguments
    # file_to_test = "test_audio/my_recording.wav"
    
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyze_file.py <path_to_audio_file>")
        print("Example: python analyze_file.py my_recording.wav")
    else:
        analyze_audio_file(sys.argv[1])