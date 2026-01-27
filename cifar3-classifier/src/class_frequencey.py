# ============================================================
# ULTRA-FAST LEARNING VERSION (2-3 min per run)
# ============================================================



from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, TrainingArguments, Trainer
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


SAMPLE_SIZE = 5000    # 10x smaller (1500 total images)
BATCH_SIZE = 128         # 2x larger (faster on GPU)
EPOCHS = 30        # Half the epochs
LEARNING_RATE = 0.00005
MODEL_NAME = "google/mobilenet_v2_1.0_224"

CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


print(f"⚡ MODE: {SAMPLE_SIZE} samples, {EPOCHS} epochs, batch {BATCH_SIZE}")


# ============================================================
# LOAD DATA (FAST)
# ============================================================

print("Loading data...")
dataset = load_dataset("cifar10")


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

    # Create augmentation pipeline
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

def transform(batch):
    # Apply augmentation only to training data
    if batch.get('is_train', False):
        imgs = [train_transforms(img) for img in batch['img']]
    else:
        imgs = batch['img']
    
    inputs = processor(imgs, return_tensors='pt')
    inputs['labels'] = batch['label']
    return inputs


# Mark training split
dataset['train'] = dataset['train'].map(lambda x: {**x, 'is_train': True})
dataset['validation'] = dataset['validation'].map(lambda x: {**x, 'is_train': False})
dataset['test'] = dataset['test'].map(lambda x: {**x, 'is_train': False})

print("Preprocessing...")
for split in ['train', 'validation', 'test']:
    dataset[split] = dataset[split].map(transform, batched=True, remove_columns=['img'])
    dataset[split].set_format('torch')

print("✓ Preprocessed")

