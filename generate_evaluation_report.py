"""
Generate comprehensive evaluation report with visualizations
Shows model performance metrics and accuracy analysis
"""

import os
import sys

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading model...")
model = tf.keras.models.load_model('CKmodel_best.h5')

print("Loading test data...")
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    'test_emotion',
    class_mode='categorical',
    target_size=(48, 48),
    shuffle=False,
    batch_size=32
)

print("\n" + "="*70)
print("EVALUATING EMOTION DETECTION MODEL")
print("="*70)

# Evaluate
print("\nEvaluating on test set...")
results = model.evaluate(val_generator, verbose=0)
print(f"\nTest Loss: {results[0]:.4f}")
print(f"Test Accuracy: {results[1]:.4f} ({results[1]*100:.2f}%)")

# Get predictions
print("\nGenerating predictions...")
predictions = model.predict(val_generator, verbose=0)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = val_generator.classes
class_names = sorted(list(val_generator.class_indices.keys()))
num_classes = len(class_names)

print(f"Predictions generated for {len(predicted_classes)} images")

# Classification report
print("\n" + "="*70)
print("CLASSIFICATION REPORT")
print("="*70)
report = classification_report(true_classes, predicted_classes, target_names=class_names, digits=3)
print(report)

# Confusion matrix
print("\n" + "="*70)
print("CONFUSION MATRIX")
print("="*70)
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

# Per-class accuracy
print("\n" + "="*70)
print("PER-CLASS ACCURACY")
print("="*70)
per_class_accuracy = []
for i, class_name in enumerate(class_names):
    mask = true_classes == i
    if mask.sum() > 0:
        class_acc = (predicted_classes[mask] == i).sum() / mask.sum()
        per_class_accuracy.append(class_acc)
        print(f"{class_name.capitalize():15} {(predicted_classes[mask] == i).sum():4d}/{mask.sum():4d} ({class_acc*100:6.2f}%)")

# Create visualizations
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# 1. Per-class accuracy bar chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Per-class accuracy
ax1 = axes[0, 0]
colors = ['#2ecc71' if acc >= 0.5 else '#e74c3c' for acc in per_class_accuracy]
ax1.barh([c.capitalize() for c in class_names], [acc*100 for acc in per_class_accuracy], color=colors, alpha=0.7, edgecolor='black')
ax1.set_xlabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
ax1.set_xlim(0, 100)
ax1.axvline(x=50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
for i, (acc, name) in enumerate(zip(per_class_accuracy, class_names)):
    ax1.text(acc*100 + 2, i, f'{acc*100:.1f}%', va='center', fontweight='bold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# Subplot 2: Confusion matrix heatmap
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, 
            ax=ax2, cbar_kws={'label': 'Count'}, square=True)
ax2.set_ylabel('True Label', fontweight='bold')
ax2.set_xlabel('Predicted Label', fontweight='bold')
ax2.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

# Subplot 3: Overall metrics
ax3 = axes[1, 0]
ax3.axis('off')
metrics_text = f"""
OVERALL METRICS

Test Accuracy: {results[1]*100:.2f}%
Test Loss: {results[0]:.4f}

Total Test Samples: {len(predicted_classes)}
Number of Classes: {num_classes}
Correct Predictions: {(predicted_classes == true_classes).sum()}
Incorrect Predictions: {(predicted_classes != true_classes).sum()}

Best Class: {class_names[np.argmax(per_class_accuracy)].capitalize()} ({max(per_class_accuracy)*100:.2f}%)
Worst Class: {class_names[np.argmin(per_class_accuracy)].capitalize()} ({min(per_class_accuracy)*100:.2f}%)
Average Accuracy: {np.mean(per_class_accuracy)*100:.2f}%
"""
ax3.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace', verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Subplot 4: Prediction distribution
ax4 = axes[1, 1]
correct = (predicted_classes == true_classes).sum()
incorrect = (predicted_classes != true_classes).sum()
colors_pie = ['#2ecc71', '#e74c3c']
wedges, texts, autotexts = ax4.pie([correct, incorrect], labels=['Correct', 'Incorrect'], 
                                     autopct='%1.1f%%', colors=colors_pie, startangle=90)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(11)
ax4.set_title(f'Overall Prediction Results\n({correct} correct, {incorrect} incorrect)', 
              fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('evaluation_report.png', dpi=300, bbox_inches='tight')
print("Saved: evaluation_report.png")
plt.close()

# 2. Detailed per-class metrics
fig, ax = plt.subplots(figsize=(12, 6))

metrics_dict = classification_report(true_classes, predicted_classes, target_names=class_names, output_dict=True)
classes_list = [c.capitalize() for c in class_names]
precision = [metrics_dict[c]['precision'] for c in class_names]
recall = [metrics_dict[c]['recall'] for c in class_names]
f1 = [metrics_dict[c]['f1-score'] for c in class_names]

x = np.arange(len(classes_list))
width = 0.25

bars1 = ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#3498db', edgecolor='black')
bars2 = ax.bar(x, recall, width, label='Recall', alpha=0.8, color='#e74c3c', edgecolor='black')
bars3 = ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#2ecc71', edgecolor='black')

ax.set_ylabel('Score', fontsize=12, fontweight='bold')
ax.set_title('Per-Class Performance Metrics (Precision, Recall, F1-Score)', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(classes_list, rotation=45, ha='right')
ax.legend(fontsize=11)
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, label='50% threshold')
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('metrics_detail.png', dpi=300, bbox_inches='tight')
print("Saved: metrics_detail.png")
plt.close()

# 3. Class distribution and prediction accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class distribution
ax1 = axes[0]
class_counts = np.bincount(true_classes)
ax1.bar([c.capitalize() for c in class_names], class_counts, color='#3498db', alpha=0.7, edgecolor='black')
ax1.set_ylabel('Number of Images', fontsize=11, fontweight='bold')
ax1.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)
for i, count in enumerate(class_counts):
    ax1.text(i, count + 10, str(count), ha='center', fontweight='bold')

# Prediction accuracy comparison
ax2 = axes[1]
accuracy_comparison = []
for i in range(num_classes):
    mask = true_classes == i
    if mask.sum() > 0:
        acc = (predicted_classes[mask] == i).sum() / mask.sum()
        accuracy_comparison.append(acc * 100)
    else:
        accuracy_comparison.append(0)

colors = ['#2ecc71' if acc >= 50 else '#e74c3c' for acc in accuracy_comparison]
bars = ax2.bar([c.capitalize() for c in class_names], accuracy_comparison, color=colors, alpha=0.7, edgecolor='black')
ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
ax2.set_title('Per-Class Prediction Accuracy', fontsize=12, fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='50% threshold')
ax2.set_ylim(0, 110)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
for bar, acc in zip(bars, accuracy_comparison):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
            f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('class_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: class_analysis.png")
plt.close()

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)
print("\nGenerated Files:")
print("  1. evaluation_report.png - Overall metrics and confusion matrix")
print("  2. metrics_detail.png - Detailed per-class performance metrics")
print("  3. class_analysis.png - Class distribution and accuracy analysis")
print("\nOpen these PNG files to view detailed visualizations of model performance.")
print("="*70)
