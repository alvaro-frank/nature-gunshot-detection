import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape):
    """
    Builds a lightweight 2D CNN for Gunshot vs Nature classification.
    
    Args:
        input_shape (tuple): The shape of the input spectrogram (e.g., (128, 87, 1))
    """
    model = models.Sequential()

    # --- Block 1: Low-level features (Edges/Lines) ---
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization()) # Helps with training stability

    # --- Block 2: Mid-level features (Textures/Patterns) ---
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.BatchNormalization())

    # --- Block 3: High-level features (Complex Shapes) ---
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # --- Classification Head ---
    model.add(layers.Flatten())
    
    # Dense layer with Dropout to prevent overfitting on your small dataset
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5)) # Randomly turns off 50% of neurons during training
    
    # Output Layer: 1 Neuron (Sigmoid) because it's Binary (0 or 1)
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model