"""
Generate graphs showing emotion and gender detection accuracy on test images
Creates two separate visualizations for model performance
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

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
emotion_predictions = defaultdict(lambda: {'correct': 0, 'total': 0})
gender_predictions = defaultdict(lambda: {'correct': 0, 'total': 0})
emotion_confidences = []
gender_confidences = []
test_image_names = []
emotion_top_predictions = defaultdict(int)
gender_top_predictions = defaultdict(int)

# Process each image
for img_path in sorted(test_images):
    filename = img_path.name
    test_image_names.append(filename.replace('.jpg', '').replace('.png', '').replace('.jpeg', ''))
    
    # Extract expected emotion from filename (e.g., "angry_man.jpg" -> "angry")
    expected_emotion = None
    for emotion_name in emotion_classes.values():
        if emotion_name.lower() in filename.lower():
            expected_emotion = emotion_name
            break
    
    # Extract expected gender from filename (e.g., "angry_man.jpg" -> "male")
    expected_gender = None
    if 'man' in filename.lower() or 'male' in filename.lower():
        expected_gender = 'male'
    elif 'woman' in filename.lower() or 'female' in filename.lower():
        expected_gender = 'female'
    
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
    emotion_idx = np.argmax(emotion_pred[0])
    emotion_confidence = emotion_pred[0][emotion_idx]
    emotion_label = emotion_classes[emotion_idx]
    
    emotion_confidences.append(emotion_confidence * 100)
    emotion_top_predictions[emotion_label] += 1
    
    if expected_emotion:
        emotion_predictions[expected_emotion]['total'] += 1
        if emotion_label.lower() == expected_emotion.lower():
            emotion_predictions[expected_emotion]['correct'] += 1
    
    # Gender detection
    gender_face = cv2.resize(face_roi_color, (96, 96))
    gender_face = gender_face.astype('float32') / 255.0
    gender_face = np.expand_dims(gender_face, axis=0)
    
    gender_pred = gender_model.predict(gender_face, verbose=0)
    gender_idx = np.argmax(gender_pred[0])
    gender_confidence = gender_pred[0][gender_idx]
    gender_label = gender_classes[gender_idx]
    
    gender_confidences.append(gender_confidence * 100)
    gender_top_predictions[gender_label] += 1
    
    if expected_gender:
        gender_predictions[expected_gender]['total'] += 1
        if gender_label.lower() == expected_gender.lower():
            gender_predictions[expected_gender]['correct'] += 1

# Calculate accuracies
print("="*80)
print("ACCURACY SUMMARY")
print("="*80)

print("\n📊 EMOTION DETECTION ACCURACY BY CLASS:")
print("-" * 40)
emotion_accuracies = []
emotion_labels = []
for emotion, stats in sorted(emotion_predictions.items()):
    if stats['total'] > 0:
        accuracy = (stats['correct'] / stats['total']) * 100
        emotion_accuracies.append(accuracy)
        emotion_labels.append(emotion.capitalize())
        print(f"{emotion.capitalize():15} {stats['correct']}/{stats['total']} correct ({accuracy:.1f}%)")

print("\n📊 GENDER DETECTION ACCURACY BY CLASS:")
print("-" * 40)
gender_accuracies = []
gender_labels = []
for gender, stats in sorted(gender_predictions.items()):
    if stats['total'] > 0:
        accuracy = (stats['correct'] / stats['total']) * 100
        gender_accuracies.append(accuracy)
        gender_labels.append(gender.capitalize())
        print(f"{gender.capitalize():15} {stats['correct']}/{stats['total']} correct ({accuracy:.1f}%)")

print("\n📊 AVERAGE CONFIDENCE SCORES:")
print("-" * 40)
print(f"Emotion: {np.mean(emotion_confidences):.2f}%")
print(f"Gender:  {np.mean(gender_confidences):.2f}%")

# Create visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Emotion Accuracy Graph
if emotion_accuracies:
    colors_emotion = ['#2ecc71' if acc >= 50 else '#e74c3c' for acc in emotion_accuracies]
    axes[0].bar(emotion_labels, emotion_accuracies, color=colors_emotion, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Emotion Detection Accuracy by Class', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 110)
    axes[0].axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (label, acc) in enumerate(zip(emotion_labels, emotion_accuracies)):
        axes[0].text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    axes[0].legend()
    axes[0].set_xticklabels(emotion_labels, rotation=45, ha='right')

# Gender Accuracy Graph
if gender_accuracies:
    colors_gender = ['#3498db', '#e91e63']
    axes[1].bar(gender_labels, gender_accuracies, color=colors_gender, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Gender Detection Accuracy by Class', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 110)
    axes[1].axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (label, acc) in enumerate(zip(gender_labels, gender_accuracies)):
        axes[1].text(i, acc + 2, f'{acc:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    axes[1].legend()
    axes[1].set_xticklabels(gender_labels, rotation=45, ha='right')

plt.tight_layout()
plt.savefig('model_accuracy_graph.png', dpi=300, bbox_inches='tight')
print("\n✅ Graph saved as 'model_accuracy_graph.png'")
plt.close()

# Create confidence distribution graph
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Emotion confidence distribution
axes[0].hist(emotion_confidences, bins=10, color='#3498db', alpha=0.7, edgecolor='black')
axes[0].axvline(np.mean(emotion_confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(emotion_confidences):.2f}%')
axes[0].set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[0].set_title('Emotion Detection Confidence Distribution', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Gender confidence distribution
axes[1].hist(gender_confidences, bins=10, color='#e91e63', alpha=0.7, edgecolor='black')
axes[1].axvline(np.mean(gender_confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(gender_confidences):.2f}%')
axes[1].set_xlabel('Confidence (%)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1].set_title('Gender Detection Confidence Distribution', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_confidence_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Graph saved as 'model_confidence_distribution.png'")
plt.close()

print("\n" + "="*80)
print("Analysis complete! Check the generated PNG files for visualizations.")
print("="*80)
