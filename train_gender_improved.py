"""
Improved gender classification model training with explicit debugging.
Uses Training/ and Validation/ folders with male/ and female/ subfolders.
Fixes the "only reports female" issue.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import json

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 20

def build_gender_model():
    """Binary gender classifier (Male/Female) with proper regularization"""
    model = keras.Sequential([
        keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        
        # Block 1
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        
        # Dense layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(2, activation='softmax')  # Male / Female
    ])
    
    return model

if __name__ == '__main__':
    print("\n" + "="*70)
    print("IMPROVED GENDER CLASSIFICATION MODEL TRAINING")
    print("="*70)
    
    # Create model
    model = build_gender_model()
    
    # Use higher learning rate with decay
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=50,
        decay_rate=0.96
    )
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Data augmentation for training (ONLY for training)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # NO augmentation for validation (just rescale)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Load training data
    print("\n" + "-"*70)
    print("Loading Training Data...")
    train_generator = train_datagen.flow_from_directory(
        'Training_gender',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['male', 'female'],  # IMPORTANT: alphabetical order
        shuffle=True
    )
    
    # Load validation data
    print("Loading Validation Data...")
    val_generator = val_datagen.flow_from_directory(
        'Validation_gender',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=['male', 'female'],  # SAME order as training
        shuffle=False
    )
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")
    
    # Calculate and use class weights to handle imbalance
    classes_unique = np.unique(train_generator.classes)
    class_weights_arr = compute_class_weight(
        'balanced',
        classes=classes_unique,
        y=train_generator.classes
    )
    class_weight = {int(c): float(w) for c, w in zip(classes_unique, class_weights_arr)}
    print(f"Class weights: {class_weight}")
    
    # Save class indices for inference
    class_indices = train_generator.class_indices
    with open('gender_class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    print(f"✓ Saved class indices to gender_class_indices.json")
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        'gender_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
    
    # Train model
    print("\n" + "-"*70)
    print(f"Training for {EPOCHS} epochs...")
    print("-"*70)
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=[early_stop, checkpoint],
        class_weight=class_weight,
        verbose=1
    )
    
    # Save final model
    model.save('gender_model.h5')
    print("\n✓ Model saved as 'gender_model.h5'")
    print("✓ Best model saved as 'gender_model_best.h5'")
    
    # Evaluate on validation set
    print("\n" + "-"*70)
    print("FINAL VALIDATION RESULTS")
    print("-"*70)
    val_loss, val_acc = model.evaluate(val_generator, verbose=0)
    print(f"Validation Accuracy: {val_acc*100:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print("\n✓ Training complete! Ready for inference.")
