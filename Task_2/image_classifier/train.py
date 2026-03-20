"""
Train an image classification model for the Animals-10 dataset.

This script reads a dataset directory where each subdirectory is an animal class,
then splits the data into train/validation/test sets and trains a transfer learning model.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np

# Ensure imports work when running from subfolders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from image_classifier.model import AnimalImageClassifier


def collect_image_paths(dataset_dir: str) -> Tuple[List[str], List[int], List[str]]:
    classes = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith('.')
    ])
    image_paths, labels = [], []

    for idx, cls in enumerate(classes):
        class_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(idx)

    return image_paths, labels, classes


def parse_args():
    parser = argparse.ArgumentParser(description="Train image classification model for Animals-10")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(PROJECT_ROOT / "animals"),
        help="Directory containing class subfolders with images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "image_classifier"),
        help="Directory to save trained model and metadata"
    )
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Fraction of data to use for training")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of data to use for validation")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="Fraction of data to use for testing")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--model-name", type=str, default="resnet50",
                        help="Model architecture to use (resnet50, resnet101, etc.)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for splits")
    parser.add_argument("--verbose", action="store_true",
                        help="Print training progress")
    return parser.parse_args()


def stratified_split(
    image_paths: List[str],
    labels: List[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
):
    """Split image paths and labels into train/val/test sets with stratification."""

    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for path, label in zip(image_paths, labels):
        grouped[label].append(path)

    train_paths, val_paths, test_paths = [], [], []
    train_labels, val_labels, test_labels = [], [], []

    for label, paths in grouped.items():
        paths = list(paths)
        rng.shuffle(paths)

        n = len(paths)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        i1 = n_train
        i2 = n_train + n_val

        train_paths.extend(paths[:i1])
        val_paths.extend(paths[i1:i2])
        test_paths.extend(paths[i2:])

        train_labels.extend([label] * n_train)
        val_labels.extend([label] * n_val)
        test_labels.extend([label] * n_test)

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels


def main():
    args = parse_args()

    image_paths, labels, class_names = collect_image_paths(args.dataset_dir)
    if not image_paths:
        raise ValueError(f"No images found in dataset directory: {args.dataset_dir}")

    train_paths, train_labels, val_paths, val_labels, test_paths, test_labels = stratified_split(
        image_paths,
        labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Total images: {len(image_paths)}")
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    classifier = AnimalImageClassifier(
        num_classes=len(class_names),
        model_name=args.model_name,
        pretrained=True,
        device=args.device
    )
    classifier.set_class_names(class_names)

    classifier.train(
        train_image_paths=train_paths,
        train_labels=train_labels,
        val_image_paths=val_paths if val_paths else None,
        val_labels=val_labels if val_labels else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        freeze_backbone=True,
        verbose=args.verbose
    )

    # Evaluate on test set if available
    if test_paths:
        loss, acc = classifier.evaluate(test_paths, test_labels, batch_size=args.batch_size)
        print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

    # Save split information
    split_info = {
        "classes": class_names,
        "train": len(train_paths),
        "val": len(val_paths),
        "test": len(test_paths)
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2)


if __name__ == "__main__":
    main()
