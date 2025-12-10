"""
Emotion Recognition Model Training Script

This script trains a deep CNN model for facial emotion classification using
Keras/TensorFlow. It loads images from directory-based datasets, applies
augmentation to training data, computes class weights for imbalanced classes,
and saves the best-performing model and class index mapping.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json

# ------------------------------------------------------------
# DATA AUGMENTATION & PREPROCESSING
# ------------------------------------------------------------

# ImageDataGenerator for training with augmentation to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Normalize pixel values
    rotation_range=15,        # Random rotation augmentation
    width_shift_range=0.1,    # Random horizontal shifting
    height_shift_range=0.1,   # Random vertical shifting
    zoom_range=0.15,          # Random zooming
    horizontal_flip=True,     # Random horizontal flipping
    fill_mode='nearest'       # Fill strategy for augmented pixels
)

# Validation generator should only rescale — no augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

# Directory paths for training and validation data
TRAINING_DIR = 'train_emotion'
VAL_DIR = 'test_emotion'

# Flow images from directories and automatically assign labels
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    class_mode='categorical',  # Multi-class classification
    target_size=(48, 48),      # Resize images
    shuffle=True               # Shuffle to ensure randomness
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    class_mode='categorical',
    target_size=(48, 48),
    shuffle=False              # Do not shuffle validation for consistency
)

# ------------------------------------------------------------
# SAVE CLASS INDICES FOR FUTURE INFERENCE
# ------------------------------------------------------------
# Mapping: class_name -> class_index
with open('class_indices.json', 'w') as f:
    json.dump({k: int(v) for k, v in train_generator.class_indices.items()}, f)

# ------------------------------------------------------------
# CALLBACKS: EARLY STOPPING & CHECKPOINTING
# ------------------------------------------------------------

# Stop training if validation accuracy stops improving
early_stop = EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=6,
    restore_best_weights=True
)

# Save only the model weights achieving highest validation accuracy
checkpoint = ModelCheckpoint(
    'emotion_model.h5',
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

callbacks = [early_stop, checkpoint]

# ------------------------------------------------------------
# MODEL ARCHITECTURE (CNN for EMOTION RECOGNITION)
# ------------------------------------------------------------

model = tf.keras.models.Sequential([
    # ------------------ Block 1 ------------------
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                           input_shape=(48, 48, 3),
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    # ------------------ Block 2 ------------------
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    # ------------------ Block 3 ------------------
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),

    # ------------------ Block 4 ------------------
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),  # Reduces parameters compared to Flatten
    tf.keras.layers.Dropout(0.4),

    # ------------------ Dense Layers ------------------
    tf.keras.layers.Dense(512, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(256, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.Dropout(0.3),

    # Final classification layer (7 emotions)
    tf.keras.layers.Dense(7, activation='softmax')
])

model.summary()  # Print model architecture

# ------------------------------------------------------------
# LEARNING RATE SCHEDULER
# ------------------------------------------------------------

# Gradually reduces LR as training progresses
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True
)

# Compile model with Adam optimizer and categorical crossentropy
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ------------------------------------------------------------
# COMPUTE CLASS WEIGHTS FOR HANDLING IMBALANCE
# ------------------------------------------------------------

classes_unique = np.unique(train_generator.classes)
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=classes_unique,
    y=train_generator.classes
)

# Convert to dictionary format required by Keras
class_weight = {int(c): float(w) for c, w in zip(classes_unique, class_weights_arr)}

# ------------------------------------------------------------
# MODEL TRAINING
# ------------------------------------------------------------

history = model.fit(
    train_generator,
    epochs=100,
    verbose=1,
    callbacks=callbacks,
    validation_data=val_generator,
    class_weight=class_weight  # Apply class weighting
)

# ------------------------------------------------------------
# SAVE FINAL MODEL
# ------------------------------------------------------------
model.save('emotion_model.h5')  # Saves complete model (architecture + weights)
