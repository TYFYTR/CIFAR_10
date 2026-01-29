# ============================================================
# ULTRA-FAST LEARNING VERSION (2-3 min per run)
# ============================================================


from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import time

# ============================================================
# CONFIG - OPTIMIZED FOR SPEED
# ============================================================
freeze_backbone = True
load_best_model_at_end = True
metric_for_best_model = "eval_accuracy"

SAMPLE_SIZE = 100               # 50x more data (critical)
BATCH_SIZE = 4                  # Larger for efficiency
EPOCHS = 1                      # More epochs with early stopping
LEARNING_RATE = 0.0001           # Keep
MODEL_NAME = "google/mobilenet_v2_1.0_224"

WEIGHT_DECAY = 0.01              # Keep (fixed from 0.1)

NUM_EPOCHS_STOP = 5              # More patience

MIXUP_ALPHA = 0.2                # Mixup augmentation alpha parameter

# Gentler augmentation
HORIZONTAL_FLIP_PROBABILITY = 0.5
RANDOM_GRAYSCALE_PROBABILITY = 0.1
ROTATION_DEGREES = 10            # Half (was too aggressive)
BRIGHTNESS = 0.2                 # Gentler
CONTRAST = 0.2                   # Gentler
SATURATION = 0.2                 # Gentler
HUE = 0.05                       # Gentler
SCALE_MIN = 0.8                  # Less aggressive crop
SCALE_MAX = 1.0

# LR Scheduler
FACTOR = 0.5
PATIENCE = 3
MIN_LR = 1e-7


NUM_CLASSES = 3                # More impressive (phase 1 baseline)
CLASSES = list(range(NUM_CLASSES))
CLASS_NAMES = None


print(f"⚡ MODE: {SAMPLE_SIZE} samples, {EPOCHS} epochs, batch {BATCH_SIZE}")


# ============================================================
# LOAD DATA (FAST)
# ============================================================

print("Loading data...")
dataset = load_dataset("ethz/food101")

# Get class names from dataset
CLASS_NAMES = dataset['train'].features['label'].names[:NUM_CLASSES]
print(f"Using classes: {CLASS_NAMES}")

# Filter dataset to only include CLASSES before sampling
print(f"Filtering dataset to classes: {CLASSES}...")
dataset['train'] = dataset['train'].filter(lambda x: x['label'] < NUM_CLASSES)
if 'validation' in dataset:
    dataset['validation'] = dataset['validation'].filter(lambda x: x['label'] < NUM_CLASSES)
if 'test' in dataset:
    dataset['test'] = dataset['test'].filter(lambda x: x['label'] < NUM_CLASSES)



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
    transforms.RandomRotation(ROTATION_DEGREES),
    transforms.ColorJitter(brightness=BRIGHTNESS, contrast=CONTRAST, saturation=SATURATION, hue=HUE),
    transforms.RandomGrayscale(p=RANDOM_GRAYSCALE_PROBABILITY),  # NEW - forces shape learning
    transforms.RandomResizedCrop(224, scale=(SCALE_MIN, SCALE_MAX)),
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

class MixupCollator:
    def __init__(self, alpha=0.2, num_classes=None, training=True):
        self.alpha = alpha
        self.num_classes = num_classes
        self.training = training
    
    def __call__(self, features):
        # Extract pixel_values and labels
        if isinstance(features[0], dict):
            pixel_values = torch.stack([f['pixel_values'] for f in features])
            labels = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        else:
            # Fallback if format is different
            pixel_values = torch.stack([f['pixel_values'] for f in features])
            labels = torch.tensor([f['labels'] for f in features], dtype=torch.long)
        
        # Only apply mixup during training
        if self.training and self.alpha > 0:
            # Generate lambda from beta distribution
            lam = np.random.beta(self.alpha, self.alpha)
            
            # Shuffle the batch to create pairs
            batch_size = pixel_values.size(0)
            indices = torch.randperm(batch_size)
            
            # Mix images
            shuffled_pixel_values = pixel_values[indices]
            mixed_pixel_values = lam * pixel_values + (1 - lam) * shuffled_pixel_values
            
            # Convert labels to one-hot encoding
            labels_onehot = torch.zeros(batch_size, self.num_classes, dtype=torch.float32)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            
            shuffled_labels_onehot = labels_onehot[indices]
            mixed_labels = lam * labels_onehot + (1 - lam) * shuffled_labels_onehot
            
            # Return mixed data with soft labels
            return {
                'pixel_values': mixed_pixel_values,
                'labels': mixed_labels
            }
        else:
            # Return original data during evaluation (hard labels)
            return {
                'pixel_values': pixel_values,
                'labels': labels
            }

def compute_metrics(pred):
    # Handle both soft labels (2D) and hard labels (1D)
    label_ids = pred.label_ids
    # Convert to numpy if tensor
    if hasattr(label_ids, 'numpy'):
        label_ids = label_ids.numpy()
    elif torch.is_tensor(label_ids):
        label_ids = label_ids.cpu().numpy()
    
    # Check if soft labels (2D one-hot)
    if len(label_ids.shape) > 1:
        # Soft labels (one-hot) - convert to hard labels by argmax
        label_ids = label_ids.argmax(axis=1)
    
    predictions = pred.predictions.argmax(-1)
    return {'accuracy': accuracy_score(label_ids, predictions)}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    

    
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none",
    load_best_model_at_end=load_best_model_at_end,
    metric_for_best_model=metric_for_best_model,
    
    weight_decay= WEIGHT_DECAY,
)

# Custom callback for ReduceLROnPlateau
class ReduceLROnPlateauCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.scheduler = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        # Initialize scheduler after optimizer is created
        self.scheduler = ReduceLROnPlateau(
            self.trainer.optimizer,
            mode='min',
            factor=FACTOR,
            patience=PATIENCE,
            min_lr=MIN_LR
        )
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if self.scheduler and logs and 'eval_loss' in logs:
            self.scheduler.step(logs['eval_loss'])
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.scheduler:
            current_lr = self.trainer.optimizer.param_groups[0]['lr']
            epoch_num = int(state.epoch) if state.epoch is not None else 0
            print(f"Epoch {epoch_num + 1} - Current LR: {current_lr:.2e}")

# Custom Trainer that uses different collators for training and evaluation
class MixupTrainer(Trainer):
    def __init__(self, train_collator=None, eval_collator=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_collator = train_collator
        self.eval_collator = eval_collator
        # Store original collator if provided
        self._original_collator = kwargs.get('data_collator', None)
    
    def get_train_dataloader(self):
        if self.train_collator is not None:
            # Temporarily replace data_collator for training
            original_collator = self.data_collator
            try:
                self.data_collator = self.train_collator
                dataloader = super().get_train_dataloader()
            finally:
                # Always restore original collator
                self.data_collator = original_collator
            return dataloader
        return super().get_train_dataloader()
    
    def get_eval_dataloader(self, eval_dataset=None):
        if self.eval_collator is not None:
            # Temporarily replace data_collator for evaluation
            original_collator = self.data_collator
            try:
                self.data_collator = self.eval_collator
                dataloader = super().get_eval_dataloader(eval_dataset)
            finally:
                # Always restore original collator
                self.data_collator = original_collator
            return dataloader
        return super().get_eval_dataloader(eval_dataset)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to handle soft labels from Mixup.
        If labels are soft (2D), use cross-entropy with soft labels (KL divergence).
        If labels are hard (1D), use standard cross-entropy.
        """
        labels = inputs.get("labels")
        
        # Check if labels are soft (2D one-hot) or hard (1D integer)
        if labels is not None and len(labels.shape) > 1:
            # Soft labels: compute loss manually using KL divergence
            # Remove labels from inputs before passing to model
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**model_inputs)
            logits = outputs.get("logits")
            # Compute soft cross-entropy: -sum(y * log(softmax(logits)))
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            loss = -torch.sum(labels * log_probs, dim=-1).mean()
            
            if return_outputs:
                return (loss, outputs)
            return loss
        else:
            # Hard labels: use standard loss computation
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

# Initialize Mixup collators
mixup_collator = MixupCollator(alpha=MIXUP_ALPHA, num_classes=NUM_CLASSES, training=True)
eval_collator = MixupCollator(alpha=0, num_classes=NUM_CLASSES, training=False)

trainer = MixupTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
    train_collator=mixup_collator,
    eval_collator=eval_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=NUM_EPOCHS_STOP)],
)

# Add ReduceLROnPlateau callback (after trainer creation so we can pass trainer reference)
trainer.add_callback(ReduceLROnPlateauCallback(trainer))

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
history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "learning_rate": []}
last_lr = LEARNING_RATE  # Track last known LR
for entry in trainer.state.log_history:
    if "loss" in entry and "eval_loss" not in entry:
        history["train_loss"].append(round(entry["loss"], 4))
    if "eval_loss" in entry:
        history["epoch"].append(entry.get("epoch", len(history["epoch"])+1))
        history["val_loss"].append(round(entry["eval_loss"], 4))
        history["val_acc"].append(round(entry.get("eval_accuracy", 0), 4))
        # Extract learning rate - check entry first, then use last known LR
        lr = entry.get("learning_rate", last_lr)
        last_lr = lr  # Update last known LR
        history["learning_rate"].append(round(lr, 8))

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
        "mixup_alpha": MIXUP_ALPHA,
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

from google.colab import files
files.download("run.json")

print(f"\n{'='*60}")
print(f"✓ Results saved to: {output_file}")
print(f"{'='*60}\n")
