import sounddevice as sd
import numpy as np

# --- SETTINGS TO TEST ---
DEVICE_ID = 1      # Try 1 first, then change to 12 (WASAPI) if 1 fails
CHANNELS = 1       # Try 1. If it crashes, try 2
SAMPLE_RATE = 44100 # Standard hardware rate (we will downsample later)

def callback(indata, frames, time, status):
    if status:
        print(f"Status Error: {status}")
    
    # Calculate volume (RMS)
    vol = np.sqrt(np.mean(indata**2))
    
    # Print a visual bar
    bar = "█" * int(vol * 100)
    print(f"Volume: {vol:.5f} | {bar}")

print(f"Testing Device {DEVICE_ID} at {SAMPLE_RATE}Hz...")

try:
    with sd.InputStream(device=DEVICE_ID, channels=CHANNELS, samplerate=SAMPLE_RATE, callback=callback):
        print("Press Ctrl+C to stop.")
        while True:
            pass
except Exception as e:
    print(f"\n❌ ERROR: {e}")