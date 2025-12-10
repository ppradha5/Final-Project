import cv2
import os
from pathlib import Path

BASE = 'faces_48'
Path(BASE).mkdir(exist_ok=True)

print('Capture utility for labeled emotion samples')
print('Instructions:')
print(' - Run this script: python capture_samples.py')
print(' - Type the emotion label (folder name) and press Enter')
print(' - Then the webcam will show. Press SPACE to capture an image, ESC or q to quit capturing for that label')
print(" - Captured images are saved into faces_48/<label>/")

while True:
    label = input('\nEnter emotion label to capture (or blank to exit): ').strip()
    if label == '':
        print('Exiting capture utility.')
        break
    label_dir = os.path.join(BASE, label)
    Path(label_dir).mkdir(parents=True, exist_ok=True)
    print(f"Starting capture for label: {label}. Press SPACE to capture, 'q' or ESC to stop")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Could not open webcam')
        break

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Frame read failed')
            break
        cv2.putText(frame, f'Label: {label}  Count: {count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.imshow('Capture - press SPACE to save, q to quit', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == 32:  # SPACE
            fname = f"{label}_{count:04d}.jpg"
            path = os.path.join(label_dir, fname)
            cv2.imwrite(path, frame)
            count += 1
            print(f'Saved {path}')
    cap.release()
    cv2.destroyAllWindows()
    print(f'Captured {count} images for label {label}')

print('Done')
