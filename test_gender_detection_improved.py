"""
Improved real-time gender detection with proper label mapping.
Fixes the "only reports female" issue.
"""

import cv2
import numpy as np
import tensorflow as tf
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load gender model
GENDER_MODEL_PATH = 'gender_model_best.h5'
if os.path.exists(GENDER_MODEL_PATH):
    print(f'Loading gender model: {GENDER_MODEL_PATH}')
    gender_model = tf.keras.models.load_model(GENDER_MODEL_PATH)
    print('✓ Gender model loaded')
else:
    print(f'ERROR: Gender model not found at {GENDER_MODEL_PATH}')
    print('Please train the model first using: python train_gender_improved.py')
    exit()

# Load class indices
if os.path.exists('gender_class_indices.json'):
    with open('gender_class_indices.json', 'r') as f:
        class_indices = json.load(f)
    print(f'✓ Loaded class indices: {class_indices}')
else:
    # Default mapping if file not found
    class_indices = {'male': 0, 'female': 1}
    print(f'Using default class indices: {class_indices}')

# Reverse mapping: index -> label
gender_labels = {v: k.upper() for k, v in class_indices.items()}
print(f'Gender labels: {gender_labels}')

print('\nStarting gender detection. Press Q to quit.')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('ERROR: Could not open webcam')
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to grab frame.')
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract face region
        crop = frame[y:y+h, x:x+w]
        try:
            crop_resized = cv2.resize(crop, (96, 96))
        except:
            continue
        
        # Normalize
        crop_norm = crop_resized.astype('float32') / 255.0
        crop_input = np.expand_dims(crop_norm, axis=0)
        
        # Predict gender
        preds = gender_model.predict(crop_input, verbose=0)
        gender_idx = int(np.argmax(preds))
        gender_conf = float(preds[0][gender_idx])
        
        # Get label using proper mapping
        gender_label = gender_labels.get(gender_idx, f'Unknown ({gender_idx})')
        
        # Display prediction
        text = f'{gender_label} ({gender_conf:.2%})'
        color = (0, 255, 0) if gender_conf > 0.7 else (0, 165, 255)
        cv2.putText(frame, f'Gender: {text}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show confidence for both classes
        male_conf = preds[0][0]
        female_conf = preds[0][1]
        cv2.putText(frame, f'Male: {male_conf:.1%}', (x, y+h+20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 100), 1)
        cv2.putText(frame, f'Female: {female_conf:.1%}', (x, y+h+40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 200), 1)
    
    # Display frame
    cv2.imshow('Gender Detection - Press Q to Quit', frame)
    
    # Quit on Q or ESC
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:
        print('Closing...')
        break
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print('Gender detection closed.')
