"""
Generate clean accuracy and confidence graphs from test image predictions
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load pre-trained models
print("Loading models...")
emotion_model = tf.keras.models.load_model('CKmodel_best.h5')
gender_model = tf.keras.models.load_model('gender_model_best.h5')

# Load class indices
with open('class_indices.json', 'r') as f:
    emotion_classes = json.load(f)
    emotion_classes = {int(v): k for k, v in emotion_classes.items()}

with open('gender_class_indices.json', 'r') as f:
    gender_classes = json.load(f)
    gender_classes = {int(v): k for k, v in gender_classes.items()}

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f"Emotion classes: {emotion_classes}")
print(f"Gender classes: {gender_classes}\n")

# Test image directory
TEST_DIR = 'test_images'
FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

# Get all test images
test_images = []
for fmt in FORMATS:
    test_images.extend(Path(TEST_DIR).glob(f'*{fmt}'))
    test_images.extend(Path(TEST_DIR).glob(f'*{fmt.upper()}'))

if not test_images:
    print(f"No images found in '{TEST_DIR}' folder.")
    exit(1)

print(f"Processing {len(test_images)} test image(s)...\n")

# Store predictions
emotion_confidences = []
gender_confidences = []

# Process each image
for img_path in sorted(test_images):
    # Read image
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        continue
    
    # Process first face found
    x, y, w, h = faces[0]
    face_roi_gray = gray[y:y+h, x:x+w]
    face_roi_color = frame[y:y+h, x:x+w]
    
    # Emotion detection
    emotion_face = cv2.resize(face_roi_gray, (48, 48))
    emotion_face = np.expand_dims(emotion_face, axis=-1)
    emotion_face = np.repeat(emotion_face, 3, axis=-1)
    emotion_face = emotion_face.astype('float32') / 255.0
    emotion_face = np.expand_dims(emotion_face, axis=0)
    
    emotion_pred = emotion_model.predict(emotion_face, verbose=0)
    emotion_confidence = np.max(emotion_pred[0]) * 100
    emotion_confidences.append(emotion_confidence)
    
    # Gender detection
    gender_face = cv2.resize(face_roi_color, (96, 96))
    gender_face = gender_face.astype('float32') / 255.0
    gender_face = np.expand_dims(gender_face, axis=0)
    
    gender_pred = gender_model.predict(gender_face, verbose=0)
    gender_confidence = np.max(gender_pred[0]) * 100
    gender_confidences.append(gender_confidence)

print(f"Collected {len(emotion_confidences)} predictions\n")

# Calculate statistics
print("="*70)
print("STATISTICS")
print("="*70)
print(f"\nEmotion Detection:")
print(f"  Average Confidence: {np.mean(emotion_confidences):.2f}%")
print(f"  Min Confidence: {np.min(emotion_confidences):.2f}%")
print(f"  Max Confidence: {np.max(emotion_confidences):.2f}%")
print(f"  Std Deviation: {np.std(emotion_confidences):.2f}%")

print(f"\nGender Detection:")
print(f"  Average Confidence: {np.mean(gender_confidences):.2f}%")
print(f"  Min Confidence: {np.min(gender_confidences):.2f}%")
print(f"  Max Confidence: {np.max(gender_confidences):.2f}%")
print(f"  Std Deviation: {np.std(gender_confidences):.2f}%")

# Create clean visualizations
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Graph 1: Accuracy Metrics Summary
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Emotion summary
ax1 = axes[0]
emotion_stats = {
    'Avg Confidence': np.mean(emotion_confidences),
    'Min': np.min(emotion_confidences),
    'Max': np.max(emotion_confidences)
}
bars1 = ax1.bar(emotion_stats.keys(), emotion_stats.values(), color=['#3498db', '#e74c3c', '#2ecc71'], 
                alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
ax1.set_title('Emotion Detection - Confidence Statistics', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars1, emotion_stats.values()):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{val:.2f}%', ha='center', fontweight='bold', fontsize=11)

# Gender summary
ax2 = axes[1]
gender_stats = {
    'Avg Confidence': np.mean(gender_confidences),
    'Min': np.min(gender_confidences),
    'Max': np.max(gender_confidences)
}
bars2 = ax2.bar(gender_stats.keys(), gender_stats.values(), color=['#9b59b6', '#e67e22', '#1abc9c'], 
                alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('Confidence (%)', fontsize=12, fontweight='bold')
ax2.set_title('Gender Detection - Confidence Statistics', fontsize=13, fontweight='bold')
ax2.set_ylim(0, 105)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars2, gender_stats.values()):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{val:.2f}%', ha='center', fontweight='bold', fontsize=11)

plt.suptitle('Model Accuracy Metrics from Test Images', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('model_accuracy_graph.png', dpi=300, bbox_inches='tight')
print("Saved: model_accuracy_graph.png")
plt.close()

# Graph 2: Confidence Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Emotion confidence distribution
ax1 = axes[0]
n1, bins1, patches1 = ax1.hist(emotion_confidences, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
ax1.axvline(np.mean(emotion_confidences), color='red', linestyle='--', linewidth=2.5, 
           label=f'Mean: {np.mean(emotion_confidences):.2f}%')
ax1.axvline(np.median(emotion_confidences), color='green', linestyle='--', linewidth=2.5, 
           label=f'Median: {np.median(emotion_confidences):.2f}%')
ax1.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax1.set_title('Emotion Detection - Confidence Distribution', fontsize=13, fontweight='bold')
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# Gender confidence distribution
ax2 = axes[1]
n2, bins2, patches2 = ax2.hist(gender_confidences, bins=15, color='#9b59b6', alpha=0.7, edgecolor='black')
ax2.axvline(np.mean(gender_confidences), color='red', linestyle='--', linewidth=2.5, 
           label=f'Mean: {np.mean(gender_confidences):.2f}%')
ax2.axvline(np.median(gender_confidences), color='green', linestyle='--', linewidth=2.5, 
           label=f'Median: {np.median(gender_confidences):.2f}%')
ax2.set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('Gender Detection - Confidence Distribution', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10, loc='upper left')
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Model Confidence Distribution Analysis', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('model_confidence_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: model_confidence_distribution.png")
plt.close()

print("\n" + "="*70)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*70)
print("\nUpdated PNG Files:")
print("  1. model_accuracy_graph.png - Clean accuracy metrics")
print("  2. model_confidence_distribution.png - Confidence histograms")
print("="*70)
