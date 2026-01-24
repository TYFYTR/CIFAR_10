# CIFAR-3 Classifier

Fine-tuning ResNet-50 on 3 CIFAR-10 classes (airplane, automobile, ship).

## Setup
```bash
pip install -r requirements.txt
```

## Train
```bash
python src/train.py
```

## Results

Check `ml_diary/cifar3/` for run snapshots.

## Lessons Learned

- Run 1: [Your notes]
- Run 2: [Your notes]
```

---

## Expected Results (Sanity Check)

**With this clean setup:**
```
Training:   85-90%
Validation: 82-88%
Test:       80-85%

Per-class: All 75-85% (balanced)
Confidence: 80-85% on correct, 50-60% on incorrect
Overfitting: <5% gap (good generalization)