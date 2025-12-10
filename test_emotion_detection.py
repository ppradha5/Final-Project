import cv2
import numpy as np
import tensorflow as tf
import json

# Load the trained model
print("Loading model...")
model = tf.keras.models.load_model('emotion_model.h5')

# Load class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Reverse the mapping to get emotion names from indices
emotion_labels = {v: k for k, v in class_indices.items()}

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Starting webcam... Press 'q' to quit")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        # Extract face region
        face_roi = frame[y:y+h, x:x+w]
        
        # Prepare image for model (48x48, grayscale)
        face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_normalized = face_resized / 255.0
        
        # Convert to 3-channel (model expects RGB)
        face_3channel = np.stack([face_normalized] * 3, axis=-1)
        
        # Add batch dimension
        face_batch = np.expand_dims(face_3channel, axis=0)
        
        # Predict emotion
        predictions = model.predict(face_batch, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx] * 100
        emotion_label = emotion_labels[emotion_idx]
        
        # Draw rectangle around face
        color = (0, 255, 0) if confidence > 70 else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        # Put emotion label and confidence
        text = f"{emotion_label}: {confidence:.1f}%"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        # Show all emotion probabilities
        y_offset = y + h + 25
        for idx, (emotion, prob) in enumerate(zip(emotion_labels.values(), predictions[0])):
            bar_text = f"{emotion}: {prob*100:.1f}%"
            cv2.putText(frame, bar_text, (x, y_offset + idx*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Display frame
    cv2.imshow('Emotion Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed")
