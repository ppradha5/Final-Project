"""
Combined real-time emotion and gender detection.
Detects both emotions (7 classes) and gender (male/female) simultaneously.
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

# Load emotion model
print("Loading emotion model...")
emotion_model = tf.keras.models.load_model('emotion_model_best.h5')

# Load emotion class indices
with open('class_indices.json', 'r') as f:
    emotion_indices = json.load(f)
emotion_labels = {v: k.upper() for k, v in emotion_indices.items()}
print(f'✓ Emotion model loaded - {len(emotion_labels)} emotions')

# Load gender model
print("Loading gender model...")
gender_model = tf.keras.models.load_model('gender_model_best.h5')

# Load gender class indices
with open('gender_class_indices.json', 'r') as f:
    gender_indices = json.load(f)
gender_labels = {v: k.upper() for k, v in gender_indices.items()}
print(f'✓ Gender model loaded - {len(gender_labels)} genders')

print('\nStarting combined emotion & gender detection.')
print('Press Q to quit.\n')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print('ERROR: Could not open webcam')
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    for (x, y, w, h) in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # ===== EMOTION DETECTION =====
        face_roi = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_resized_emotion = cv2.resize(face_gray, (48, 48))
        face_norm_emotion = face_resized_emotion / 255.0
        face_3channel = np.stack([face_norm_emotion] * 3, axis=-1)
        face_batch_emotion = np.expand_dims(face_3channel, axis=0)
        
        emotion_predictions = emotion_model.predict(face_batch_emotion, verbose=0)
        emotion_idx = np.argmax(emotion_predictions[0])
        emotion_conf = emotion_predictions[0][emotion_idx] * 100
        emotion_label = emotion_labels[emotion_idx]
        
        # ===== GENDER DETECTION =====
        face_resized_gender = cv2.resize(face_roi, (96, 96))
        face_norm_gender = face_resized_gender.astype('float32') / 255.0
        face_batch_gender = np.expand_dims(face_norm_gender, axis=0)
        
        gender_predictions = gender_model.predict(face_batch_gender, verbose=0)
        gender_idx = int(np.argmax(gender_predictions[0]))
        gender_conf = gender_predictions[0][gender_idx] * 100
        gender_label = gender_labels[gender_idx]
        
        # ===== DISPLAY RESULTS =====
        # Top line: Gender and Emotion
        emotion_color = (0, 255, 0) if emotion_conf > 70 else (0, 165, 255)
        gender_color = (0, 255, 0) if gender_conf > 70 else (0, 165, 255)
        
        cv2.putText(frame, f'Gender: {gender_label} ({gender_conf:.1f}%)', 
                   (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gender_color, 2)
        cv2.putText(frame, f'Emotion: {emotion_label} ({emotion_conf:.1f}%)', 
                   (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
        
        # Gender probabilities
        y_offset = y + h + 20
        cv2.putText(frame, f'Male: {gender_predictions[0][0]*100:.1f}%', 
                   (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 200), 1)
        cv2.putText(frame, f'Female: {gender_predictions[0][1]*100:.1f}%', 
                   (x, y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 100, 200), 1)
        
        # Emotion probabilities (all 7 emotions)
        y_offset = y + h + 50
        for emotion_id in range(len(emotion_labels)):
            emotion_name = emotion_labels[emotion_id]
            emotion_score = emotion_predictions[0][emotion_id] * 100
            cv2.putText(frame, f'{emotion_name}: {emotion_score:.1f}%', 
                       (x, y_offset + emotion_id*18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 100), 1)
    
    # Display frame
    cv2.imshow('Emotion & Gender Detection - Press Q to Quit', frame)
    
    # Quit on Q or ESC
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == ord('Q') or key == 27:
        print('\nClosing detection...')
        break
    
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print('✓ Detection closed.')
