import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the best trained model
model = tf.keras.models.load_model('CKmodel_best.h5')

# Create validation data generator (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

VAL_DIR = 'test_emotion'

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    class_mode='categorical',
    target_size=(48, 48),
    shuffle=False,
    batch_size=32
)

# Evaluate the model
print("Evaluating model on test set...")
results = model.evaluate(val_generator, verbose=1)

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Test Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")
print("="*50)

# Get predictions for detailed analysis
print("\nGetting predictions on test set...")
predictions = model.predict(val_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes

# Calculate accuracy per class
from sklearn.metrics import classification_report, confusion_matrix
class_names = list(val_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_names))

print("\nConfusion Matrix:")
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)
