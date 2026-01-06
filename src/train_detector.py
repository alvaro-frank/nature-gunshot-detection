import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Import the model structure we just created
from model import build_model

# --- CONFIGURATION ---
DATA_DIR = "processed_data"
IMG_HEIGHT = 128  # Mel bands
IMG_WIDTH = 87    # Time steps (approximate for 2s @ 22050Hz)
BATCH_SIZE = 32
EPOCHS = 20

def load_data():
    """
    Loads .npy files from processed_data/gunshots and processed_data/nature_noise
    Returns: X (features), y (labels)
    """
    X = []
    y = []
    
    # Label 0: Nature (The Negative Class)
    # Label 1: Gunshot (The Positive Class)
    categories = {"nature_noise": 0, "gunshots": 1}
    
    print("Loading data...")
    for category, label in categories.items():
        dir_path = os.path.join(DATA_DIR, category)
        if not os.path.exists(dir_path):
            print(f"Error: Directory {dir_path} not found!")
            continue
            
        files = [f for f in os.listdir(dir_path) if f.endswith('.npy')]
        for f in files:
            # Load the spectrogram
            spectrogram = np.load(os.path.join(dir_path, f))
            
            # Ensure shape consistency (sometimes rounding errors occur in preprocessing)
            # We crop or pad to ensure exactly IMG_WIDTH
            if spectrogram.shape[1] > IMG_WIDTH:
                spectrogram = spectrogram[:, :IMG_WIDTH]
            elif spectrogram.shape[1] < IMG_WIDTH:
                padding = IMG_WIDTH - spectrogram.shape[1]
                spectrogram = np.pad(spectrogram, ((0,0), (0, padding)))
                
            X.append(spectrogram)
            y.append(label)
            
    print(f"Loaded {len(X)} total samples.")
    return np.array(X), np.array(y)

def main():
    # 1. Load Data
    X, y = load_data()
    
    # 2. Reshape for CNN (Add channel dimension)
    # Current shape: (N, 128, 87) -> New shape: (N, 128, 87, 1)
    X = X[..., np.newaxis]
    
    # 3. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Handle Imbalance (The 851 vs 280 problem)
    # This calculates how much weight to give 'Nature' vs 'Gunshot'
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    weights_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"\nClass Weights applied: {weights_dict}")
    # Likely result: {0: ~2.5, 1: ~0.6} (Nature counts 4x more than gunshot)

    # 5. Build Model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 1)
    model = build_model(input_shape)
    model.summary()
    
    # 6. Train
    print("\nStarting Training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        class_weight=weights_dict  # <--- Critical for your dataset
    )
    
    # 7. Evaluate
    print("\nEvaluating on Test Set...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # Confusion Matrix (To see if we are confusing twigs for guns)
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    print("\n[[True Nature, False Alarm (Type I)]")
    print(" [Missed Shot (Type II), True Gunshot]]")
    
    # 8. Save Model
    model.save("models/forest_gunshot_detector.keras")
    print("\nModel saved as 'models/forest_gunshot_detector.keras'")

if __name__ == "__main__":
    main()