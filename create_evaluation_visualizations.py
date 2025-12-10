"""
Generate evaluation visualizations based on actual model accuracy data
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Actual test results data
emotions = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
correct = [310, 69, 56, 1208, 610, 449, 685]
total = [958, 111, 1024, 1774, 1233, 1247, 831]
accuracy = [32.36, 62.16, 5.47, 68.09, 49.47, 36.01, 82.43]

# Calculate total accuracy
total_correct = sum(correct)
total_samples = sum(total)
overall_accuracy = (total_correct / total_samples) * 100

print("="*70)
print("EMOTION DETECTION MODEL - EVALUATION REPORT")
print("="*70)
print(f"\nOverall Accuracy: {overall_accuracy:.2f}%")
print(f"Total Correct: {total_correct}/{total_samples}")
print("\nPer-Class Accuracy:")
for i, emotion in enumerate(emotions):
    print(f"  {emotion:12} {correct[i]:4d}/{total[i]:4d} ({accuracy[i]:6.2f}%)")
print("="*70)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Per-class accuracy bar chart (large)
ax1 = fig.add_subplot(gs[0, :])
colors = ['#2ecc71' if acc >= 50 else '#e74c3c' for acc in accuracy]
bars = ax1.bar(emotions, accuracy, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_title('Per-Class Emotion Detection Accuracy', fontsize=14, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.axhline(y=50, color='orange', linestyle='--', linewidth=2, alpha=0.6, label='50% threshold')
ax1.axhline(y=overall_accuracy, color='blue', linestyle='-', linewidth=2, alpha=0.6, label=f'Overall: {overall_accuracy:.2f}%')
ax1.grid(axis='y', alpha=0.3)
ax1.legend(fontsize=11, loc='upper left')

# Add value labels on bars
for bar, acc, corr, tot in zip(bars, accuracy, correct, total):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{acc:.2f}%\n({corr}/{tot})', ha='center', va='bottom', 
            fontsize=10, fontweight='bold')

# 2. Confusion indicators (predictions vs ground truth)
ax2 = fig.add_subplot(gs[1, 0])
correct_pct = [c/t*100 for c, t in zip(correct, total)]
incorrect_pct = [100 - c for c in correct_pct]

x_pos = np.arange(len(emotions))
width = 0.6

bars1 = ax2.bar(x_pos, correct_pct, width, label='Correct', color='#2ecc71', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x_pos, incorrect_pct, width, bottom=correct_pct, label='Incorrect', color='#e74c3c', alpha=0.8, edgecolor='black')

ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax2.set_title('Prediction Correctness by Class', fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(emotions, rotation=45, ha='right')
ax2.set_ylim(0, 100)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)

# 3. Test samples distribution
ax3 = fig.add_subplot(gs[1, 1])
colors_dist = plt.cm.Set3(np.linspace(0, 1, len(emotions)))
bars = ax3.barh(emotions, total, color=colors_dist, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Number of Samples', fontsize=11, fontweight='bold')
ax3.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Add value labels
for bar, tot in zip(bars, total):
    ax3.text(tot + 20, bar.get_y() + bar.get_height()/2.,
            f'{tot}', va='center', fontweight='bold', fontsize=10)

# 4. Summary metrics table
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

# Create summary text
summary_text = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     EMOTION DETECTION MODEL EVALUATION                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  OVERALL PERFORMANCE:                                                        ║
║    Total Accuracy:        {overall_accuracy:.2f}%                                           ║
║    Correct Predictions:   {total_correct:,}/{total_samples:,} samples                                ║
║                                                                              ║
║  BEST PERFORMING CLASS:                                                      ║
║    Emotion:               {emotions[np.argmax(accuracy)]} ({max(accuracy):.2f}%)                                ║
║                                                                              ║
║  WORST PERFORMING CLASS:                                                     ║
║    Emotion:               {emotions[np.argmin(accuracy)]} ({min(accuracy):.2f}%)                                  ║
║                                                                              ║
║  CLASS PERFORMANCE SUMMARY:                                                  ║
║    Excellent (≥70%):      {len([a for a in accuracy if a >= 70])} classes ({', '.join([emotions[i] for i, a in enumerate(accuracy) if a >= 70])})                 ║
║    Good (50-70%):         {len([a for a in accuracy if 50 <= a < 70])} classes ({', '.join([emotions[i] for i, a in enumerate(accuracy) if 50 <= a < 70])})             ║
║    Poor (<50%):           {len([a for a in accuracy if a < 50])} classes ({', '.join([emotions[i] for i, a in enumerate(accuracy) if a < 50])})              ║
║                                                                              ║
║  INSIGHTS:                                                                   ║
║    • Surprised faces detected with very high accuracy (82.43%)               ║
║    • Happy faces detected well (68.09%), strong positive emotion detection   ║
║    • Fearful faces challenging (5.47%) - easily confused with other emotions║
║    • Angry and Sad faces underperforming - similar facial expressions       ║
║    • Neutral emotion moderate performance (49.47%)                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9.5,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=1))

plt.suptitle('Real-Time Face Analysis System - Emotion Detection Evaluation', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('emotion_accuracy_report.png', dpi=300, bbox_inches='tight')
print("\nSaved: emotion_accuracy_report.png")
plt.close()

# Create detailed per-class analysis
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, (emotion, corr, tot, acc) in enumerate(zip(emotions, correct, total, accuracy)):
    ax = axes[idx]
    
    # Create pie chart for each class
    sizes = [corr, tot - corr]
    colors_pie = ['#2ecc71', '#e74c3c']
    explode = (0.05, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, labels=['Correct', 'Incorrect'], 
                                        autopct='%1.1f%%', colors=colors_pie, 
                                        explode=explode, startangle=90)
    
    # Beautify text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    for text in texts:
        text.set_fontsize(9)
        text.set_fontweight('bold')
    
    # Add title with accuracy and samples
    title_color = '#2ecc71' if acc >= 50 else '#e74c3c'
    ax.set_title(f'{emotion}\n{acc:.2f}% ({corr}/{tot})', 
                fontsize=11, fontweight='bold', color=title_color, pad=10)

# Hide the last empty subplot
axes[-1].axis('off')

plt.suptitle('Per-Class Emotion Detection Performance', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('emotion_per_class_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: emotion_per_class_analysis.png")
plt.close()

# Create comparison chart
fig, ax = plt.subplots(figsize=(12, 6))

# Sort by accuracy for visualization
sorted_indices = np.argsort(accuracy)[::-1]
sorted_emotions = [emotions[i] for i in sorted_indices]
sorted_accuracy = [accuracy[i] for i in sorted_indices]
sorted_correct = [correct[i] for i in sorted_indices]
sorted_total = [total[i] for i in sorted_indices]

colors_sorted = ['#2ecc71' if acc >= 50 else '#e74c3c' for acc in sorted_accuracy]

bars = ax.barh(sorted_emotions, sorted_accuracy, color=colors_sorted, alpha=0.8, edgecolor='black', linewidth=2)

ax.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Emotion Detection Accuracy (Ranked)', fontsize=13, fontweight='bold')
ax.set_xlim(0, 100)
ax.axvline(x=50, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='50% threshold')
ax.axvline(x=overall_accuracy, color='blue', linestyle='-', linewidth=2, alpha=0.5, label=f'Overall: {overall_accuracy:.2f}%')
ax.legend(fontsize=11)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar, acc, corr, tot in zip(bars, sorted_accuracy, sorted_correct, sorted_total):
    ax.text(acc + 1, bar.get_y() + bar.get_height()/2.,
           f'{acc:.2f}% ({corr}/{tot})', va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('emotion_ranked_accuracy.png', dpi=300, bbox_inches='tight')
print("Saved: emotion_ranked_accuracy.png")
plt.close()

print("\n" + "="*70)
print("VISUALIZATION GENERATION COMPLETE!")
print("="*70)
print("\nGenerated Files:")
print("  1. emotion_accuracy_report.png - Comprehensive evaluation report")
print("  2. emotion_per_class_analysis.png - Pie charts for each emotion class")
print("  3. emotion_ranked_accuracy.png - Ranked accuracy comparison")
print("\nAll visualizations are ready for presentation to your professor!")
print("="*70)
