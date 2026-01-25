"""
CIFAR-3 Classifier Training Script
===================================
Fine-tune ResNet-50 on 3 CIFAR-10 classes.

Usage: python src/train.py
"""

from datasets import load_dataset
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    TrainingArguments, 
    Trainer
)
from sklearn.metrics import accuracy_score
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_NAME = "cifar3_classifier"
MODEL_NAME = "google/efficientnet-b0"
CLASSES_TO_USE = [0, 1, 8]  # airplane, automobile, ship
CLASS_NAMES = ["airplane", "automobile", "ship"]

HYPERPARAMS = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "num_epochs": 10,
    "eval_strategy": "epoch",
}

# ============================================================
# 1. LOAD DATA
# ============================================================

print("Loading CIFAR-10 dataset...")
dataset = load_dataset("cifar10")

# Filter to 3 classes
def filter_classes(example):
    return example['label'] in CLASSES_TO_USE

dataset = dataset.filter(filter_classes)

# Remap labels 0,1,8 → 0,1,2
label_map = {CLASSES_TO_USE[i]: i for i in range(len(CLASSES_TO_USE))}
def remap_labels(example):
    example['label'] = label_map[example['label']]
    return example

dataset = dataset.map(remap_labels)

# Split
train_val = dataset['train'].train_test_split(test_size=0.2, seed=42)
dataset = {
    'train': train_val['train'],
    'validation': train_val['test'],
    'test': dataset['test']
}

print(f"Train: {len(dataset['train'])}")
print(f"Validation: {len(dataset['validation'])}")
print(f"Test: {len(dataset['test'])}")

# ============================================================
# 2. LOAD MODEL
# ============================================================

print(f"\nLoading {MODEL_NAME}...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    ignore_mismatched_sizes=True
)

# ============================================================
# 3. PREPROCESS
# ============================================================

def transform(batch):
    inputs = processor(batch['img'], return_tensors='pt')
    inputs['labels'] = batch['label']
    return inputs

print("Preprocessing...")
for split in ['train', 'validation', 'test']:
    dataset[split] = dataset[split].map(
        transform, 
        batched=True, 
        remove_columns=['img']
    )
    dataset[split].set_format('torch')

# ============================================================
# 4. TRAINING SETUP
# ============================================================

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {'accuracy': accuracy_score(labels, preds)}

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=HYPERPARAMS['num_epochs'],
    per_device_train_batch_size=HYPERPARAMS['batch_size'],
    per_device_eval_batch_size=HYPERPARAMS['batch_size'],
    learning_rate=HYPERPARAMS['learning_rate'],
    eval_strategy=HYPERPARAMS['eval_strategy'],
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='./logs',
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    compute_metrics=compute_metrics,
)

# ============================================================
# 5. TRAIN
# ============================================================

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

trainer.train()

# ============================================================
# 6. EVALUATE
# ============================================================

print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60 + "\n")

train_results = trainer.evaluate(dataset['train'])
val_results = trainer.evaluate(dataset['validation'])
test_results = trainer.evaluate(dataset['test'])

print(f"Training:   {train_results['eval_accuracy']:.1%}")
print(f"Validation: {val_results['eval_accuracy']:.1%}")
print(f"Test:       {test_results['eval_accuracy']:.1%}")

# ============================================================
# 7. ANALYSIS
# ============================================================

print("\n" + "="*60)
print("DETAILED ANALYSIS")
print("="*60 + "\n")

from analyze import run_analysis

run_analysis(
    trainer=trainer,
    dataset=dataset,
    class_names=CLASS_NAMES,
    save_dir="./plots"
)

# ============================================================
# 8. SAVE TO DIARY
# ============================================================

from ml_diary import save_run_snapshot

save_run_snapshot(
    project_name=PROJECT_NAME,
    model_name=MODEL_NAME,
    dataset_info={
        "source": "CIFAR-10",
        "classes": CLASS_NAMES,
        "train_size": len(dataset['train']),
        "test_size": len(dataset['test'])
    },
    hyperparameters=HYPERPARAMS,
    results={
        "train_acc": train_results['eval_accuracy'],
        "val_acc": val_results['eval_accuracy'],
        "test_acc": test_results['eval_accuracy'],
        "train_val_gap": train_results['eval_accuracy'] - val_results['eval_accuracy']
    },
    notes="First clean run with good data. [Add your observations here]",
    confusion_matrix_path="./plots/confusion_matrix.png"
)

print("\n✅ Training complete! Check ml_diary/ for snapshot.")