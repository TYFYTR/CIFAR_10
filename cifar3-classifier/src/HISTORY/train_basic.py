# ============================================================
# ULTRA-FAST LEARNING VERSION (2-3 min per run)
# ============================================================



from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report
import torch
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time

# ============================================================
# CONFIG - OPTIMIZED FOR SPEED
# ============================================================
freeze_backbone = True           # NEW
load_best_model_at_end = True    # NEW
metric_for_best_model = "eval_accuracy"


SAMPLE_SIZE = 200   # 10x smaller (1500 total images)
BATCH_SIZE = 32        # 2x larger (faster on GPU)
EPOCHS = 1        # Half the epochs
LEARNING_RATE = 0.003
MODEL_NAME = "google/mobilenet_v2_1.0_224"

CLASSES = [0, 1, 8]
CLASS_NAMES = ["airplane", "automobile", "ship"]



print(f"⚡ MODE: {SAMPLE_SIZE} samples, {EPOCHS} epochs, batch {BATCH_SIZE}")


# ============================================================
# LOAD DATA (FAST)
# ============================================================

print("Loading data...")
dataset = load_dataset("cifar10")
dataset = dataset.filter(lambda x: x['label'] in CLASSES)

label_map = {CLASSES[i]: i for i in range(len(CLASSES))}
dataset = dataset.map(lambda x: {'label': label_map[x['label']]})

# Take small sample
dataset['train'] = dataset['train'].shuffle(seed=42).select(range(SAMPLE_SIZE))
dataset['test'] = dataset['test'].shuffle(seed=42).select(range(90))  # Small test set

train_val = dataset['train'].train_test_split(test_size=0.2, seed=42)
dataset = {
    'train': train_val['train'],
    'validation': train_val['test'],
    'test': dataset['test']
}

print(f"✓ Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")

# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading {MODEL_NAME}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CLASSES), ignore_mismatched_sizes=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using: {device.upper()}")

# ============================================================
# PREPROCESS (FAST)
# ============================================================

def transform(batch):
    inputs = processor(batch['img'], return_tensors='pt')
    inputs['labels'] = batch['label']
    return inputs

print("Preprocessing...")
for split in ['train', 'validation', 'test']:
    dataset[split] = dataset[split].map(transform, batched=True, remove_columns=['img'])
    dataset[split].set_format('torch')

print("✓ Preprocessed")

# ============================================================
# TRAINING SETUP
# ============================================================

def compute_metrics(pred):
    return {'accuracy': accuracy_score(pred.label_ids, pred.predictions.argmax(-1))}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    eval_strategy="epoch",
    save_strategy="no",  # Don't save checkpoints (faster)
    logging_steps=10,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
)

# ============================================================
# TRAIN (FAST!)
# ============================================================

print("\n" + "="*60)
print("⚡ TRAINING")
print("="*60 + "\n")

start_time = time.time()
trainer.train()
training_time = (time.time() - start_time) / 60

# ============================================================
# EVALUATE
# ============================================================

print("\n" + "="*60)
print("📊 RESULTS")
print("="*60 + "\n")

train_results = trainer.evaluate(dataset['train'])
val_results = trainer.evaluate(dataset['validation'])
test_results = trainer.evaluate(dataset['test'])

print(f"Training:   {train_results['eval_accuracy']:.1%}")
print(f"Validation: {val_results['eval_accuracy']:.1%}")
print(f"Test:       {test_results['eval_accuracy']:.1%}")
print(f"Time:       {training_time:.1f} minutes ⚡")
print(f"Gap:        {(train_results['eval_accuracy'] - val_results['eval_accuracy']):.1%}")

# Quick interpretation
gap = train_results['eval_accuracy'] - val_results['eval_accuracy']
if gap > 0.15:
    print("\n⚠️  OVERFITTING detected (>15% gap)")
elif gap > 0.05:
    print("\n⚠️  Slight overfitting (5-15% gap)")
else:
    print("\n✓ Good generalization (<5% gap)")


# ============================================================
# CAPTURE FOR CURSOR (COMPLETE)
# ============================================================
import json
import torch.nn.functional as F
import torch
from sklearn.metrics import classification_report

# Extract training history
history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": []}
for entry in trainer.state.log_history:
    if "loss" in entry and "eval_loss" not in entry:
        history["train_loss"].append(round(entry["loss"], 4))
    if "eval_loss" in entry:
        history["epoch"].append(entry.get("epoch", len(history["epoch"])+1))
        history["val_loss"].append(round(entry["eval_loss"], 4))
        history["val_acc"].append(round(entry.get("eval_accuracy", 0), 4))

# Get predictions with confidence
predictions = trainer.predict(dataset['test'])
y_pred = predictions.predictions.argmax(-1)
y_true = predictions.label_ids
logits = torch.tensor(predictions.predictions)
probs = F.softmax(logits, dim=1)
confidences = probs.max(dim=1).values.tolist()

# Confidence breakdown
correct_conf = [confidences[i] for i in range(len(y_pred)) if y_pred[i] == y_true[i]]
incorrect_conf = [confidences[i] for i in range(len(y_pred)) if y_pred[i] != y_true[i]]

# Per-class metrics
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

# Calculate confusion matrix
predictions = trainer.predict(dataset['test'])
y_pred = predictions.predictions.argmax(-1)
y_true = predictions.label_ids
cm = confusion_matrix(y_true, y_pred)

results = {
    "config": {
        "samples": SAMPLE_SIZE,
        "batch": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "model": MODEL_NAME,
        "classes": CLASS_NAMES,
    },
    "metrics": {
        "train": round(train_results['eval_accuracy'], 4),
        "val": round(val_results['eval_accuracy'], 4),
        "test": round(test_results['eval_accuracy'], 4),
        "gap": round(train_results['eval_accuracy'] - val_results['eval_accuracy'], 4),
        "time_min": round(training_time, 2),
    },
    "per_class": {
        name: {
            "accuracy": round(cm[i][i]/cm[i].sum(), 4),
            "precision": round(report[name]["precision"], 4),
            "recall": round(report[name]["recall"], 4),
            "f1": round(report[name]["f1-score"], 4),
            "support": int(report[name]["support"]),
        } for i, name in enumerate(CLASS_NAMES)
    },
    "confidence": {
        "correct_mean": round(sum(correct_conf)/len(correct_conf), 4) if correct_conf else 0,
        "incorrect_mean": round(sum(incorrect_conf)/len(incorrect_conf), 4) if incorrect_conf else 0,
        "correct_dist": [round(c, 3) for c in correct_conf],
        "incorrect_dist": [round(c, 3) for c in incorrect_conf],
    },
    "confusion_matrix": cm.tolist(),
    "history": history,
    "predictions": {
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "confidences": [round(c, 4) for c in confidences],
    },
}

with open("run.json", "w") as f:
    json.dump(results, f)

from google.colab import files
files.download("run.json")