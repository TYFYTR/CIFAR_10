# ============================================================
# ULTRA-FAST LEARNING VERSION (2-3 min per run)
# ============================================================


from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
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


SAMPLE_SIZE = 100
BATCH_SIZE = 4       # 2x larger (faster on GPU)
EPOCHS = 1           # Half the epochs
LEARNING_RATE = 0.0001
MODEL_NAME = "google/mobilenet_v2_1.0_224"

WEIGHT_DECAY = 0.01

NUM_EPOCHS_STOP = 3 

WARMUP_STEPS = 100

HORIZONTAL_FLIP_PROBABILITY = 0.5
ROTATION_DEGREES = 20
BRIGHTNESS = 0.3
CONTRAST = 0.3
SATURATION = 0.3
HUE = 0.1
SCALE_MIN = 0.7
SCALE_MAX = 1.0
TRANSLATE_MIN = 0.1
TRANSLATE_MAX = 0.1


CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49]
CLASS_NAMES = ["apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare", "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito", "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake", "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla", "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder", "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes", "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict", "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras", "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice", "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",]


print(f"⚡ MODE: {SAMPLE_SIZE} samples, {EPOCHS} epochs, batch {BATCH_SIZE}")


# ============================================================
# LOAD DATA (FAST)
# ============================================================

print("Loading data...")
dataset = load_dataset("ethz/food101")

# Filter dataset to only include CLASSES before sampling
print(f"Filtering dataset to classes: {CLASSES}...")
dataset['train'] = dataset['train'].filter(lambda x: x['label'] in CLASSES)
if 'validation' in dataset:
    dataset['validation'] = dataset['validation'].filter(lambda x: x['label'] in CLASSES)
if 'test' in dataset:
    dataset['test'] = dataset['test'].filter(lambda x: x['label'] in CLASSES)

# Take small sample
dataset['train'] = dataset['train'].shuffle(seed=42).select(range(SAMPLE_SIZE))

# Create train / val / test split from train
train_temp = dataset['train'].train_test_split(test_size=0.2, seed=42)
val_test = train_temp['test'].train_test_split(test_size=0.5, seed=42)

dataset = {
    'train': train_temp['train'],
    'validation': val_test['train'],
    'test': val_test['test']
}

print(f"✓ Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}, Test: {len(dataset['test'])}")


# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading {MODEL_NAME}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME, num_labels=len(CLASSES), ignore_mismatched_sizes=True # change number to amoutn of classes
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using: {device.upper()}")



# ============================================================
# PREPROCESS (FAST)
# ============================================================


# Create augmentation pipeline
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=HORIZONTAL_FLIP_PROBABILITY),
    transforms.RandomRotation(ROTATION_DEGREES),  # Food can be at any angle
    transforms.ColorJitter(brightness=BRIGHTNESS, contrast=CONTRAST, saturation=SATURATION, hue=HUE),  # Lighting varies!
    transforms.RandomResizedCrop(224, scale=(SCALE_MIN, SCALE_MAX)),  # Zoom in/out
    transforms.RandomAffine(degrees=0, translate=(TRANSLATE_MIN, TRANSLATE_MAX)),  # Slight shifts
])

def transform(batch):
    from PIL import Image
    
    imgs = batch['image']
    
    # Apply augmentation only to training data
    if batch.get('is_train', False):
        imgs = [train_transforms(img.convert('RGB')) for img in imgs]
    else:
        imgs = [img.convert('RGB') for img in imgs]
    
    # Process images
    inputs = processor(imgs, return_tensors='pt')
    
    # Labels should just be the batch labels (already 0-indexed if filtered correctly)
    inputs['labels'] = batch['label']
    
    return inputs

# Mark training split
dataset['train'] = dataset['train'].map(lambda x: {**x, 'is_train': True})
dataset['validation'] = dataset['validation'].map(lambda x: {**x, 'is_train': False})
dataset['test'] = dataset['test'].map(lambda x: {**x, 'is_train': False})

print("Preprocessing...")
for split in ['train', 'validation', 'test']:
    dataset[split] = dataset[split].map(transform, batched=True, remove_columns=['image'])
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
    weight_decay= WEIGHT_DECAY,

        # LR Scheduler
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,

    eval_strategy="epoch",
    save_strategy="epoch",  # Don't save checkpoints (faster)
    logging_steps=10,
    report_to="none",
    #label_smoothing_factor=LABEL_SMOOTHING_FACTOR,
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model=metric_for_best_model,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=NUM_EPOCHS_STOP)],
)

# ============================================================
# Check point 
# ============================================================


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
report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASSES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

# Calculate confusion matrix
predictions = trainer.predict(dataset['test'])
y_pred = predictions.predictions.argmax(-1)
y_true = predictions.label_ids
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))

results = {
    "config": {
        "samples": SAMPLE_SIZE,
        "batch": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "model": MODEL_NAME,
        "classes": CLASS_NAMES,
        "weight_decay": WEIGHT_DECAY,
        "label_smoothing_factor": LABEL_SMOOTHING_FACTOR,
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
            "accuracy": round(cm[i][i]/cm[i].sum(), 4) if cm[i].sum() else 0,
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

# Save results to JSON file
output_file = "run.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"✓ Results saved to: {output_file}")
print(f"{'='*60}\n")
