import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
import queue
import time
import soundfile as sf

# --- CONFIGURATION ---
MODEL_PATH = "models/forest_gunshot_detector.keras"
HARDWARE_RATE = 44100
SAMPLE_RATE = 22050
DURATION = 2.0        # Seconds of audio to analyze per step
HOP_LENGTH = 0.5      # How often to check (overlap). 0.5s means we check every half second.
THRESHOLD = 0.85      # 85% confidence required to trigger
debug_frames = []

# Load Model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded. Warming up...")

# Audio Buffer (Queue)
# We keep a rolling buffer of the last 2 seconds
buffer_size = int(SAMPLE_RATE * DURATION)
audio_buffer = np.zeros(buffer_size)

def process_audio(audio_data):
    """
    Takes raw audio (2 seconds), converts to spectrogram, and predicts.
    """
    # 1. Preprocess (Same as training)
    mels = librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE, n_mels=128)
    mels_db = librosa.power_to_db(mels, ref=np.max)
    
    # 2. Reshape for Model (1, 128, 87, 1)
    # Ensure time-width is correct (87 columns)
    target_width = 87
    if mels_db.shape[1] < target_width:
        mels_db = np.pad(mels_db, ((0,0), (0, target_width - mels_db.shape[1])))
    else:
        mels_db = mels_db[:, :target_width]
        
    spectrogram = mels_db.reshape(1, 128, target_width, 1)
    
    # 3. Predict
    prediction = model.predict(spectrogram, verbose=0)[0][0]
    return prediction

def audio_callback(indata, frames, time, status):
    """
    This function is called by the microphone thread every time new audio comes in.
    """
    global audio_buffer
    if status:
        print(status)
        
    debug_frames.append(indata.copy())
    
    # Flatten incoming audio
    new_data = indata.flatten()
    
    # Shift buffer to the left and append new data (Rolling Window)
    audio_buffer = np.roll(audio_buffer, -len(new_data))
    audio_buffer[-len(new_data):] = new_data

def main():
    # Helper to print colored text
    RED = '\033[91m'
    GREEN = '\033[92m'
    RESET = '\033[0m'

    print("\n--- LISTENING STARTED ---")
    print(f"Sensitivity Threshold: {THRESHOLD * 100}%")
    print("Press Ctrl+C to stop.\n")

    # Start Recording Stream
    # blocksize=int(SAMPLE_RATE * HOP_LENGTH) means we get a callback every 0.5 seconds
    with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, 
                        blocksize=int(SAMPLE_RATE * HOP_LENGTH)):
        
        while True:
            try:
                # Analyze the current buffer (last 2 seconds)
                confidence = process_audio(audio_buffer)
                
                # Visual Feedback
                if confidence > THRESHOLD:
                    print(f"{RED}● GUNSHOT DETECTED! ({confidence*100:.1f}%){RESET}")
                else:
                    # Print a subtle heartbeat so you know it's working
                    # \r overwrites the line to keep terminal clean
                    print(f"{GREEN}○ Listening... (Noise level: {confidence*100:.1f}%){RESET}", end='\r')
                
                # Sleep briefly to sync with the hop length roughly
                # (The stream runs in a separate thread, so this loop just handles inference)
                time.sleep(HOP_LENGTH)
                
            except KeyboardInterrupt:
                print("\nStopped.")
                full_audio = np.concatenate(debug_frames, axis=0)
                sf.write('debug_output.wav', full_audio, HARDWARE_RATE)
                print("Saved 'debug_output.wav'. Run analyze_file.py on this file!")
                break

if __name__ == "__main__":
    main()