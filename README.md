# Real-Time Face Analysis System

A deep learning-based system for real-time emotion and gender detection using convolutional neural networks (CNN).

## 📋 Project Overview

This system uses two trained CNN models to detect:
- **Emotion**: 7 classes (angry, disgusted, fearful, happy, neutral, sad, surprised)
- **Gender**: 2 classes (male, female)

The models are trained on facial image datasets and can perform inference in real-time on webcam feeds or static images.

## 🎯 Model Performance

| Model | Task | Accuracy | Test Samples |
|-------|------|----------|--------------|
| emotion_model.h5 | Emotion Detection | 47.19% | 7,178 |
| gender_model_best.h5 | Gender Detection | ~85-90% | 58,658 |

### Emotion Class Distribution
- Happy: 75% precision
- Neutral: 45% precision
- Sad: 42% precision
- Angry: 38% precision
- Fearful: 5% recall (challenging class)
- Disgusted: Limited training samples
- Surprised: Limited training samples

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow opencv-python numpy scikit-learn
```

### Option 1: Live Webcam Detection
```bash
python main.py
```
This will start real-time emotion and gender detection on your webcam.
- Press **Q** to quit

### Option 2: Test on Static Images (Removed)
Static image testing has been removed. Use the webcam flow:
```bash
python main.py
```

## 📁 Project Structure

### Core Files
- **`main.py`** - Main entry point for live webcam detection
- **`test_combined_detection.py`** - Combined emotion+gender detection implementation
- **`emotion_model.py`** - Emotion model training script
- **`train_gender_improved.py`** - Gender model training script
- **`evaluate_model.py`** - Evaluation script for emotion model

### Pre-trained Models
- **`emotion_model.h5`** - Trained emotion detection model
- **`gender_model_best.h5`** - Trained gender detection model
- **`class_indices.json`** - Emotion class mapping
- **`gender_class_indices.json`** - Gender class mapping

### Data
- **`train_emotion/`** - Training images organized by emotion (7 subdirectories)
- **`test_emotion/`** - Test images organized by emotion (7 subdirectories)
- **`Training_gender/`** - Gender training data (male/, female/)
- **`Validation_gender/`** - Gender validation data (male/, female/)

## 🏗️ Architecture

### Emotion Detection Model (CKmodel_best.h5)
```
Input: 48x48 RGB images

Block 1: Conv(64) → BatchNorm → Conv(64) → BatchNorm → MaxPool → Dropout(0.25)
Block 2: Conv(128) → BatchNorm → Conv(128) → BatchNorm → MaxPool → Dropout(0.25)
Block 3: Conv(256) → BatchNorm → Conv(256) → BatchNorm → MaxPool → Dropout(0.25)
Block 4: Conv(512) → BatchNorm → GlobalAvgPool → Dropout(0.4)

Dense: 512 → BatchNorm → Dropout(0.4) → 256 → Dropout(0.3) → 7 (softmax)

Output: 7-class emotion probabilities
```

### Gender Detection Model (gender_model_best.h5)
```
Input: 96x96 RGB images

Block 1: Conv(32) → BatchNorm → MaxPool → Dropout(0.25)
Block 2: Conv(64) → BatchNorm → MaxPool → Dropout(0.25)
Block 3: Conv(128) → BatchNorm → GlobalAvgPool → Dropout(0.4)

Dense: 256 → Dropout(0.3) → 2 (softmax)

Output: 2-class gender probabilities (male/female)
```

## 🔧 Technical Details

### Optimization Strategy
- **Optimizer**: Adam with ExponentialDecay learning rate schedule
- **Initial Learning Rate**: 1e-3 (emotion), decays by 0.96 every 100 steps
- **Loss Function**: Categorical crossentropy
- **Regularization**: L2 penalties (1e-4) on all layers
- **Early Stopping**: Patience=6 (no improvement threshold)
- **Class Weighting**: Balanced weights to handle class imbalance

### Data Augmentation
**Training Data**:
- Rotation: 15°
- Width/height shift: 10%
- Zoom: 15%
- Horizontal flip: Yes

**Validation Data**: Rescaling only (no augmentation)

### Face Detection
- Haar Cascade classifier (`haarcascade_frontalface_default.xml`)
- Detects frontal faces in images/video

## 📊 Output Format

### Live Detection Output
```
Emotion: Happy (92.34%)
Gender: Female (88.21%)

Emotion probabilities:
- Happy: 92.34%
- Neutral: 4.23%
- Sad: 2.10%
- Angry: 1.12%
- Fearful: 0.15%
- Surprised: 0.04%
- Disgusted: 0.02%
### Static Image Output (Removed)
Removed in webcam-only mode.
      - Male: 11.79%
```

## 🔄 Retraining Models (Optional)

### Train Emotion Model
```bash
python emotion_model.py
```
- Requires: `train_emotion/` and `test_emotion/` directories with emotion subdirectories
- Output: `emotion_model.h5` (best model saved automatically)
- Training time: ~10-20 minutes

### Train Gender Model
```bash
python train_gender_improved.py
```
- Requires: `Training_gender/` and `Validation_gender/` directories with gender subdirectories
- Output: `gender_model_best.h5`
- Training time: ~5-10 minutes

### Evaluate Emotion Model
```bash
python evaluate_model.py
```
- Tests on full test set with detailed per-class metrics
- Outputs confusion matrix and accuracy breakdown

## ⚙️ System Requirements

- **Python**: 3.8+
- **TensorFlow**: 2.x
- **OpenCV**: 4.x
- **NumPy**: 1.19+
- **scikit-learn**: For class weight calculation
- **Webcam**: For live detection (optional, not needed for static images)
- **GPU**: Optional (will use CPU if unavailable)

## 🐛 Troubleshooting

**Issue**: "No faces detected"
- Solution: Ensure good lighting and face is clearly visible

**Issue**: Models not loading
- Solution: Check that `CKmodel_best.h5` and `gender_model_best.h5` exist in project directory

**Issue**: Webcam not working
- Solution: Check camera permissions and that no other app is using the camera

**Issue**: Low accuracy on certain emotions
- Solution: Fearful, disgusted, and surprised classes have limited training samples. Consider collecting more data for these classes.

## 📝 Limitations & Future Improvements

### Current Limitations
- Emotion detection accuracy is moderate (47.19%) - some emotions are challenging
- Works best with frontal faces
- Single face processing at a time (in static images)
- Requires good lighting conditions

### Potential Improvements
- Use transfer learning (VGG16, ResNet50, MobileNet) for better accuracy
- Implement age detection model
- Multi-face simultaneous detection
- Head pose estimation for profile faces
- Integration with emotion-based applications

## 📚 References

- Dataset: CK+ (Cohn-Kanade) and FER2013 for emotion
- Framework: TensorFlow/Keras
- Face Detection: OpenCV Haar Cascades

## 👨‍💼 For Professors/Evaluators

To test the system:

1. **Live Demo**:
   ```bash
   python main.py
2. **Model Evaluation**:
   ```bash
   python evaluate_model.py
   ```
   - Add 7-10 test images
   - Run `python test_on_images.py`

3. **Model Evaluation**:
   ```bash
   python evaluate_model.py
   ```

All outputs include confidence scores and probability distributions for transparency.

---

**Project Status**: ✅ Complete and ready for evaluation

**Last Updated**: December 8, 2025
