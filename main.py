"""
Real-Time Face Analysis System
Main entry point for emotion and gender detection using pre-trained models

This project uses deep convolutional neural networks to detect:
- Emotion: 7 classes (angry, disgusted, fearful, happy, neutral, sad, surprised)
- Gender: 2 classes (male, female)

USAGE:
Simply run: python main.py

The system will start real-time detection on your webcam.
Press 'Q' to quit.

REQUIREMENTS:
- TensorFlow 2.x
- OpenCV (cv2)
- NumPy
- Pre-trained models: CKmodel_best.h5, gender_model_best.h5

"""

import sys

def main():
    """Run real-time emotion and gender detection"""
    print("="*70)
    print("Real-Time Face Analysis System")
    print("="*70)
    print("\nStarting real-time emotion and gender detection...")
    print("Press 'Q' to quit\n")
    
    try:
        from test_combined_detection import main as detection_main
        detection_main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure both pre-trained models exist:")
        print("  ✓ CKmodel_best.h5 (emotion detector)")
        print("  ✓ gender_model_best.h5 (gender detector)")
        sys.exit(1)

if __name__ == '__main__':
    main()
