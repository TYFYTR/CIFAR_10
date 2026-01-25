import json
import matplotlib.pyplot as plt
import numpy as np

def load(path="results/run1.json"):
    with open(path) as f:
        return json.load(f)

def summary(r):
    m = r["metrics"]
    print(f"\n{'='*60}")
    print(f"TEST: {m['test']:.1%} | GAP: {m['gap']:.1%} | TIME: {m['time_min']:.1f}m")
    print(f"{'='*60}")
    
    # Diagnosis
    if m["gap"] > 0.15:
        print("⚠️  OVERFITTING: Train-val gap >15%. Need regularization.")
    elif m["gap"] < 0.02 and m["test"] < 0.85:
        print("⚠️  UNDERFITTING: Low gap but low accuracy. Need more capacity/data.")
    else:
        print("✓  Generalization looks reasonable.")

def per_class(r):
    print(f"\n{'='*60}")
    print("PER-CLASS BREAKDOWN")
    print(f"{'='*60}")
    print(f"{'Class':<12} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-" * 48)
    for name, stats in r["per_class"].items():
        print(f"{name:<12} {stats['accuracy']:>7.1%} {stats['precision']:>7.1%} {stats['recall']:>7.1%} {stats['f1']:>7.1%}")
    
    # Find weakest class
    weakest = min(r["per_class"].items(), key=lambda x: x[1]["f1"])
    print(f"\n⚠️  Weakest: {weakest[0]} (F1: {weakest[1]['f1']:.1%})")

def confidence_analysis(r):
    c = r["confidence"]
    print(f"\n{'='*60}")
    print("CONFIDENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Correct predictions:   {c['correct_mean']:.1%} avg confidence")
    print(f"Incorrect predictions: {c['incorrect_mean']:.1%} avg confidence")
    
    gap = c["correct_mean"] - c["incorrect_mean"]
    if gap < 0.1:
        print("⚠️  Model is overconfident on wrong predictions. Poor calibration.")
    else:
        print(f"✓  Confidence gap: {gap:.1%} (model knows when it's unsure)")

def learning_curve(r, save_dir="plots"):
    h = r["history"]
    if not h["val_acc"]:
        print("No history data.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Loss curve
    axes[0].plot(h["val_loss"], 'b-', label="val_loss", linewidth=2)
    if h["train_loss"]:
        # Align train_loss to epochs (it logs more frequently)
        epochs = len(h["val_loss"])
        train_sampled = h["train_loss"][::max(1, len(h["train_loss"])//epochs)][:epochs]
        axes[0].plot(train_sampled, 'r--', label="train_loss", alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(h["val_acc"], 'b-', marker='o', label="val_acc", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].grid(True, alpha=0.3)
    
    # Confidence distribution
    c = r["confidence"]
    if c["correct_dist"]:
        axes[2].hist(c["correct_dist"], bins=15, alpha=0.6, label="Correct", color="green")
    if c["incorrect_dist"]:
        axes[2].hist(c["incorrect_dist"], bins=15, alpha=0.6, label="Incorrect", color="red")
    axes[2].set_xlabel("Confidence")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Confidence Distribution")
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/analysis1.png", dpi=150)
    print(f"\nSaved: {save_dir}/analysis1.png")

def confusion(r, save_dir="plots"):
    cm = np.array(r["confusion_matrix"])
    classes = r["config"]["classes"]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (Test Acc: {r['metrics']['test']:.1%})")
    
    # Add numbers
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", 
                   color="white" if cm[i,j] > cm.max()/2 else "black")
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/confusion1.png", dpi=150)
    print(f"Saved: {save_dir}/confusion1.png")

def diagnose(r):
    """Print actionable diagnosis."""
    print(f"\n{'='*60}")
    print("DIAGNOSIS & NEXT STEPS")
    print(f"{'='*60}")
    
    m = r["metrics"]
    c = r["confidence"]
    
    issues = []
    
    if m["gap"] > 0.15:
        issues.append(("OVERFITTING", "Try: dropout, data augmentation, fewer epochs, weight decay"))
    
    if m["test"] < 0.80:
        issues.append(("LOW ACCURACY", "Try: more data, larger model, lower learning rate"))
    
    if c["incorrect_mean"] > 0.7:
        issues.append(("OVERCONFIDENT", "Try: label smoothing, temperature scaling, mixup"))
    
    weakest = min(r["per_class"].items(), key=lambda x: x[1]["f1"])
    if weakest[1]["f1"] < 0.7:
        issues.append((f"WEAK CLASS: {weakest[0]}", "Check: class imbalance, hard examples, augmentation"))
    
    if issues:
        for issue, fix in issues:
            print(f"\n⚠️  {issue}")
            print(f"   → {fix}")
    else:
        print("\n✓ Model looks solid. Consider: more data or harder task.")

if __name__ == "__main__":
    r = load()
    summary(r)
    per_class(r)
    confidence_analysis(r)
    diagnose(r)
    learning_curve(r)
    confusion(r)