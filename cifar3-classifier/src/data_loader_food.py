import tensorflow_datasets as tfds
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
# CONFIG - Number of classes to use
# ============================================================
NUM_CLASSES = 10  # Change this to use different number of classes (max 101)

print(f"Loading Food-101 using TensorFlow Datasets...")
print(f"Using first {NUM_CLASSES} classes\n")

try:
    # Load Food-101 dataset using tfds.image_classification.Food101
    # Access the builder through tfds
    builder = tfds.image_classification.Food101()
    builder.download_and_prepare()
    dataset = builder.as_dataset(as_supervised=True)
    info = builder.info
    
    # Get all class names from dataset info
    all_class_names = info.features['label'].names
    
    # Filter to selected number of classes
    class_names = all_class_names[:NUM_CLASSES]
    
    print(f"✓ Dataset loaded successfully!")
    print(f"✓ Total available classes: {len(all_class_names)}")
    print(f"✓ Using classes: {len(class_names)}")
    print(f"✓ Total train samples: {info.splits['train'].num_examples}")
    print(f"✓ Total test samples: {info.splits['test'].num_examples}")
    
    # Print selected class names
    print(f"\n{'='*60}")
    print(f"SELECTED CLASS NAMES (first {NUM_CLASSES}):")
    print(f"{'='*60}")
    for idx, name in enumerate(class_names):
        print(f"{idx:3d}: {name}")
    
    # Filter dataset to only include selected classes
    print(f"\n{'='*60}")
    print(f"Filtering dataset to first {NUM_CLASSES} classes...")
    print(f"{'='*60}")
    
    def filter_classes(image, label):
        """Filter function to keep only samples with labels < NUM_CLASSES"""
        return label < NUM_CLASSES
    
    # Filter train and test datasets
    train_ds = dataset['train'].filter(filter_classes)
    test_ds = dataset['test'].filter(filter_classes)
    
    # Count samples in filtered datasets
    train_count = sum(1 for _ in train_ds)
    test_count = sum(1 for _ in test_ds)
    
    print(f"✓ Filtered train samples: {train_count}")
    print(f"✓ Filtered test samples: {test_count}")
    
    # Print sample data
    print(f"\n{'='*60}")
    print("SAMPLE DATA:")
    print(f"{'='*60}")
    for image, label in train_ds.take(1):
        label_np = int(label.numpy())
        print(f"Label: {label_np} ({class_names[label_np]})")
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        break
    
    # Count samples per class in filtered train set
    print(f"\n{'='*60}")
    print(f"SAMPLE COUNTS PER CLASS (in filtered dataset):")
    print(f"{'='*60}")
    class_counts = {i: 0 for i in range(NUM_CLASSES)}
    for image, label in train_ds:
        label_np = int(label.numpy())
        if label_np < NUM_CLASSES:
            class_counts[label_np] += 1
    
    for idx in range(NUM_CLASSES):
        print(f"Class {idx:3d} ({class_names[idx]:<30}): {class_counts[idx]:5d} samples")
    
    # Store filtered datasets for use
    filtered_dataset = {
        'train': train_ds,
        'test': test_ds,
        'class_names': class_names,
        'num_classes': NUM_CLASSES
    }
    
    print(f"\n✓ Dataset ready! Use 'filtered_dataset' dict to access train/test splits and class names")
    
except Exception as e:
    print(f"\n❌ Error loading dataset: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()

