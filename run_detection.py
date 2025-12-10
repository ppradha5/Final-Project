import cv2
import numpy as np
import tensorflow as tf
import os
import sys
import json

# Suppress TensorFlow verbose output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load trained emotion model (prefer transfer model if available)
model_path = 'CKmodel_transfer.h5' if os.path.exists('CKmodel_transfer.h5') else 'CKmodel.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f'{model_path} not found. Run training first.')

model = tf.keras.models.load_model(model_path)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(f'Loaded model: {model_path}')

# Emotion labels - prefer class_indices file saved during training
default_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
emotion_labels = default_labels
try:
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    # invert mapping to get index->label in correct order
    idx_to_label = {int(v): k for k, v in class_indices.items()}
    emotion_labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    print('Loaded class indices mapping for labels:', emotion_labels)
except Exception:
    print('class_indices.json not found; using default label order.')

# If model output size differs, create generic labels
if model.output_shape[-1] != len(emotion_labels):
    # generate generic label names based on number
    emotion_labels = [f'emotion_{i}' for i in range(model.output_shape[-1])]

print(f'Starting webcam with {model_path}. Press Q to quit.')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    # Fallback to default backend if DirectShow fails
    cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Could not open webcam (VideoCapture returned false). Try a different camera index or backend.')

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to grab frame. Exiting.')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
        crop = frame[y:y+h, x:x+w]
        try:
            # Use 96x96 for transfer model, 48x48 for basic CNN
            size = 96 if 'transfer' in model_path else 48
            # If basic CNN, use grayscale to match common training config
            if size == 48:
                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                crop_resized = cv2.resize(crop_gray, (size, size))
                crop_resized = np.expand_dims(crop_resized, axis=-1)
            else:
                crop_resized = cv2.resize(crop, (size, size))
        except Exception:
            continue
        crop_resized = crop_resized.astype('float32') / 255.0
        crop_resized = np.expand_dims(crop_resized, axis=0)

        # Predict with verbose=0 to suppress output
        preds = model.predict(crop_resized, verbose=0)
        # Softmax is returned; show top-2 to avoid collapsing to few labels
        top2_idx = np.argsort(preds[0])[::-1][:2]
        top2 = [(emotion_labels[int(i)], float(preds[0][int(i)])) for i in top2_idx]
        label, confidence = top2[0]

        # Add confidence score to label
        if len(top2) > 1:
            display_text = f'{top2[0][0]} ({top2[0][1]:.2f}) | {top2[1][0]} ({top2[1][1]:.2f})'
        else:
            display_text = f'{label} ({confidence:.2f})'
        cv2.putText(frame, display_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow(f'Emotion Detection - {model_path} (Press Q to quit)', frame)
    
    # Better keyboard handling - longer waitKey and check for both 'q' and 'Q'
    key = cv2.waitKey(30) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print('Quit signal received. Closing...')
        break
    elif key == 27:  # ESC key as alternative exit
        print('ESC pressed. Closing...')
        break
    
    frame_count += 1
    if frame_count % 30 == 0:
        print(f'Running... Press Q to quit. (Frame {frame_count})')

cap.release()
cv2.destroyAllWindows()
print('Detection closed.')
