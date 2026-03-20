# Task 2. Animal Verification Pipeline: NER + Image Classification

A comprehensive machine learning pipeline that combines Named Entity Recognition (NER) and Image Classification to verify if a text description matches an image containing an animal.

## Overview

This project implements a multi-modal verification system with two core components:

1. **NER Component**: Extracts animal entity mentions from text using transformer-based models
2. **Image Classification Component**: Identifies which animal is present in an image using transfer learning
3. **Unified Pipeline**: Compares text and image to provide boolean verification (True/False)

## Architecture

### Pipeline Flow

```
INPUT:
  - Text: "There is a dog in the picture"
  - Image: [224x224 RGB tensor]

NER STAGE:
  - Model: BERT + Token Classification
  - Extracts: ["dog"]
  - Confidence: 0.95

IMAGE CLASSIFICATION STAGE:
  - Model: ResNet50 (transfer learning)
  - Predicts: "dog"
  - Confidence: 0.87

MATCHING STAGE:
  - Compare: ["dog"] vs "dog"
  - Fuzzy matching enabled
  - Result: MATCH ✓

OUTPUT:
  - Boolean: True
  - Confidence: 0.87
```

## Components

### 1. NER Model (`ner/model.py`)

**Purpose**: Extract animal entity mentions from text

**Architecture**:
- Base model: BERT (bert-base-uncased)
- Task: Token classification
- Labels: O (outside), B-ANIMAL (begin animal), I-ANIMAL (inside animal)

**Key Features**:
- IOB tagging format
- Fine-tuning on animal entity extraction
- Confidence scoring
- Batch prediction support

**Usage**:
```python
from ner.model import AnimalNER

ner = AnimalNER()
result = ner.predict("There is a cat in the room")
print(result["entity_types"])  # ['cat']
```

### 2. Image Classification Model (`image_classifier/model.py`)

**Purpose**: Classify which animal is in an image

**Architecture**:
- Base model: ResNet50 (pretrained on ImageNet)
- Transfer learning with fine-tuning
- 15 animal classes by default (customizable)

**Key Features**:
- Data augmentation (rotation, flip, color jitter)
- Validation during training
- Batch prediction
- Top-K predictions with confidence scores

**Preprocessing**:
- Resize: 224x224
- Normalize: ImageNet mean/std
- Augmentation: Random rotation (±20°), horizontal flip, color jitter

**Usage**:
```python
from image_classifier.model import AnimalImageClassifier

classifier = AnimalImageClassifier(num_classes=15)
result = classifier.predict("path/to/image.jpg", top_k=5)
print(result["top_prediction"])  # {'class': 'cat', 'confidence': 0.92}
```

### 3. Unified Pipeline (`pipeline.py`)

**Purpose**: Combine NER and image classification for verification

**Features**:
- Single API for verification
- Fuzzy matching for entity-class comparison
- Confidence thresholds
- Batch processing
- Detailed result logging

**Verification Logic**:
1. Extract animal entities from text using NER
2. Classify image using image classifier
3. Match extracted entities with predicted class
4. Apply fuzzy matching for variations (cat/kitten, dog/puppy, etc.)
5. Return boolean result with confidence

**Usage**:
```python
from pipeline import AnimalVerificationPipeline

pipeline = AnimalVerificationPipeline()
result = pipeline.verify(
    text="There is a dog in the picture",
    image_path="path/to/image.jpg",
    return_details=True
)
print(result["match"])  # True/False
```

## Dataset Preparation

### Expected Dataset Structure

This repository uses the Animals-10 dataset from Kaggle:
https://www.kaggle.com/datasets/alessiocorrado99/animals10

- Download and unzip the dataset.
- Rename the top-level folder to `data/animals` (or create a symlink).
- Ensure the class folders are normalized to clean names (e.g. `cat`, `dog`, `horse`, etc.)

```
data/
├── animals/
│   ├── cat/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── dog/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── horse/
│   │   └── ...
│   └── ... (10 classes from Animals-10)
└── annotations/
    ├── ner_train.json
    └── image_splits.json
```

## Installation

### Requirements
- Python 3.9-3.11

### Setup

1. Clone repository and navigate to Task_2:
   ```bash
   cd Task_2
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Exploratory Data Analysis

Open and run the Jupyter notebooks:
```bash
jupyter notebook eda.ipynb
jupyter notebook demo.ipynb
```

The notebooks include:
- Dataset statistics and distribution
- Class balance analysis
- Sample image visualization
- NER synthetic data generation
- Data splitting strategy
- Pipeline integration testing
- Demo for pipeline inference and edge cases (empty text, missing image, mismatch)


### 2. Train Models

#### Train NER Model (CLI)
Use the animals list from the dataset directory or provide your own list.

```bash
python train/train_ner.py --animals-dir ./animals --output-dir ./models/ner_model --epochs 3 --batch-size 8
```

If you have a labeled training file (JSONL with `text` and `entities` fields):

```bash
python train/train_ner.py --train-data data/ner_train.jsonl --output-dir ./models/ner_model
```

#### Train Image Classifier (CLI)
Train on the Animals-10 dataset directory structure (one folder per class):

```bash
python train/train_image_classifier.py --dataset-dir ./animals --output-dir ./models/image_classifier --epochs 10 --batch-size 32
```

If you prefer to use the Python API directly, the same workflow can be done via `image_classifier/model.py`.

### 3. Inference Scripts

Run inference without writing any code:

```bash
python infer/infer_ner.py --model-dir ./models/ner_model --text "There is a dog in the picture"
python infer/infer_image_classifier.py --model-dir ./models/image_classifier --image ./animals/dog/0001.jpg
```

### 4. Use Pipeline

**Single Verification**:
```python
from pipeline import AnimalVerificationPipeline

pipeline = AnimalVerificationPipeline(
    ner_model_path="./models/ner_model",
    image_classifier_path="./models/image_classifier"
)

result = pipeline.verify(
    text="There is a lion in the picture",
    image_path="data/animals/lion/image1.jpg",
    return_details=True
)

print(f"Match: {result['match']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Reason: {result['reason']}")
```

**Batch Verification**:
```python
text_image_pairs = [
    ("There is a dog", "path/to/dog.jpg"),
    ("I see a cat", "path/to/cat.jpg"),
    ("There is a bird", "path/to/dog.jpg"),  # Should return False
]

results = pipeline.batch_verify(text_image_pairs, return_details=True)
for result in results:
    print(f"Match: {result['match']}, Reason: {result['reason']}")
```

### 4. Run Main Pipeline Script

```bash
python pipeline.py \
  --text "There is a dog in the picture" \
  --image ./animals/dog/0001.jpg \
  --ner-model ./models/ner_model \
  --image-model ./models/image_classifier \
  --print-details
```


## Performance Considerations

### NER Model
- **Training Time**: 5-10 minutes (with 1000 samples)
- **Inference Time**: ~100-200ms per text
- **Memory**: ~2GB GPU memory

### Image Classifier
- **Training Time**: 1-3 hours (with 10,000 images per class)
- **Inference Time**: ~50-100ms per image
- **Memory**: ~4GB GPU memory

### Full Pipeline
- **Verification Time**: ~200-300ms per text-image pair
- **Total Memory**: ~6GB GPU memory

## Fuzzy Matching Examples

The pipeline includes fuzzy matching for common animal variations:

| Extracted | Predicted | Match |
|-----------|-----------|-------|
| dog | dog | ✓ |
| dog | puppy | ✓ |
| cat | kitten | ✓ |
| bird | eagle | ✓ |
| bird | sparrow | ✓ |
| horse | pony | ✓ |
| dog | cat | ✗ |

## Files Structure

```
Task_2/
├── eda.ipynb                          # Exploratory Data Analysis notebook
├── ner/                               # NER model + scripts
│   ├── model.py                       # NER model implementation
│   ├── train.py                       # NER training script
│   └── infer.py                       # NER inference script
├── image_classifier/                  # Image classification model + scripts
│   ├── model.py                       # Image classification implementation
│   ├── train.py                       # Image classification training script
│   └── infer.py                       # Image classification inference script
├── pipeline.py                        # Unified verification pipeline
├── animals/                           # Animal image dataset (Animals-10)
├── models/                            # Saved trained model checkpoints
│   ├── ner_model/                     # Trained NER model files
│   └── image_classifier/              # Trained image classifier files
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Environment Variables

Optional environment variables for configuration:

```bash
# GPU settings
export CUDA_VISIBLE_DEVICES=0

# Model paths
export NER_MODEL_PATH="./models/ner_model"
export IMAGE_CLASSIFIER_PATH="./models/image_classifier"

# Thresholds
export NER_THRESHOLD=0.5
export IMAGE_THRESHOLD=0.5
```

