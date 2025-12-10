"""
Quick script to check emotion detection per-class accuracy on the validation set.
Useful after training completes to see improvement on all 7 emotions.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

MODEL_PATH = 'CKmodel.h5'
BEST_MODEL_PATH = 'CKmodel.h5'
VAL_DIR = 'test'
IMG_SIZE = 48
BATCH_SIZE = 64
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

print(f"\nLoading best model from {BEST_MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    model_used = BEST_MODEL_PATH
except:
    print(f"  {BEST_MODEL_PATH} not found, trying {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    model_used = MODEL_PATH

print(f"✓ Model loaded: {model_used}\n")

# Load validation data
val_aug = ImageDataGenerator(rescale=1./255)
color_mode = 'rgb'
val_gen = val_aug.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode=color_mode,
    class_mode='categorical',
    classes={e: i for i, e in enumerate(EMOTIONS)},
    shuffle=False
)

print(f"Evaluating on {val_gen.samples} validation samples...\n")
steps = math.ceil(val_gen.samples / val_gen.batch_size)
preds = model.predict(val_gen, steps=steps, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes[:len(y_pred)]

# Per-emotion statistics
from collections import defaultdict
counts = defaultdict(int)
correct = defaultdict(int)

for t, p in zip(y_true, y_pred):
    counts[t] += 1
    if t == p:
        correct[t] += 1

print("="*60)
print("PER-EMOTION ACCURACY")
print("="*60)
total_correct = 0
total_count = 0
for i, emotion in enumerate(EMOTIONS):
    corr = correct[i]
    total = counts[i]
    acc = corr / total if total > 0 else 0.0
    total_correct += corr
    total_count += total
    bar_len = int(acc * 30)
    bar = '█' * bar_len + '░' * (30 - bar_len)
    print(f"{emotion:12s} {acc:.1%} [{bar}] ({corr}/{total})")

overall = total_correct / total_count if total_count > 0 else 0.0
print("="*60)
print(f"OVERALL ACCURACY: {overall:.1%}")
print("="*60 + "\n")
