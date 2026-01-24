"""
Model Analysis Tools
====================
Confusion matrix, per-class metrics, confidence analysis.
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import torch
import torch.nn.functional as F
import os

def run_analysis(trainer, dataset, class_names, save_dir="./plots"):
    """Run complete analysis and save plots."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get predictions
    predictions = trainer.predict(dataset['test'])
    y_pred = predictions.predictions.argmax(-1)
    y_true = predictions.label_ids
    
    # 1. Confusion Matrix
    print("1. Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=150)
    plt.close()
    print(f"   Saved: {save_dir}/confusion_matrix.png")
    
    # Print breakdown
    for i, true_class in enumerate(class_names):
        for j, pred_class in enumerate(class_names):
            count = cm[i][j]
            if count > 0:
                if i == j:
                    print(f"   ✓ {true_class}: {count} correct")
                else:
                    print(f"   ✗ {true_class} → {pred_class}: {count} mistakes")
    
    # 2. Per-class metrics
    print("\n2. Per-Class Performance:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=3)
    print(report)
    
    # 3. Confidence analysis
    print("\n3. Confidence Distribution:")
    logits = torch.tensor(predictions.predictions)
    probs = F.softmax(logits, dim=1)
    
    correct_conf = []
    incorrect_conf = []
    
    for idx in range(len(y_pred)):
        conf = probs[idx][y_pred[idx]].item()
        if y_pred[idx] == y_true[idx]:
            correct_conf.append(conf)
        else:
            incorrect_conf.append(conf)
    
    if len(correct_conf) > 0:
        print(f"   Correct predictions: {sum(correct_conf)/len(correct_conf):.1%} avg confidence")
    if len(incorrect_conf) > 0:
        print(f"   Incorrect predictions: {sum(incorrect_conf)/len(incorrect_conf):.1%} avg confidence")
    
    # Plot confidence distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(correct_conf) > 0:
        ax.hist(correct_conf, bins=20, alpha=0.6, label='Correct', color='green')
    if len(incorrect_conf) > 0:
        ax.hist(incorrect_conf, bins=20, alpha=0.6, label='Incorrect', color='red')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Confidence Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{save_dir}/confidence_distribution.png', dpi=150)
    plt.close()
    print(f"   Saved: {save_dir}/confidence_distribution.png")
    
    print("\n✅ Analysis complete!")