"""
Train a Named Entity Recognition (NER) model for animal entity extraction.

This script can generate synthetic training data from a list of animal classes or
load a JSONL training file with structure:
  {"text": "...", "entities": ["cat", "dog"]}

The trained model is saved to a directory that can be loaded for inference.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path so scripts can run from subfolders.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ner.model import AnimalNER


def load_animals_from_dir(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and not d.startswith('.')
    ])


def load_training_data_from_jsonl(path: str):
    texts, entities = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            texts.append(record.get('text', ""))
            entities.append(record.get('entities', []))
    return texts, entities


def generate_synthetic_data(animals: List[str], num_samples: int = 500):
    templates = [
        "There is a {} in the picture.",
        "I can see a {} here.",
        "This image shows a {}.",
        "I spotted a {} in the photo.",
        "The {} is visible in this image.",
        "I found a {} in the picture.",
        "This is a picture of a {}.",
        "There is a beautiful {} here.",
        "A {} can be seen in the image.",
        "I see a {} in this photo.",
    ]

    texts, entities = [], []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        animal = animals[i % len(animals)]
        texts.append(template.format(animal))
        entities.append([animal])

    return texts, entities


def parse_args():
    parser = argparse.ArgumentParser(description="Train NER model for animal extraction")
    parser.add_argument(
        "--animals-dir",
        type=str,
        default=str(PROJECT_ROOT / "animals"),
        help="Directory containing class subfolders (used to infer animal names)"
    )
    parser.add_argument("--animals", type=str, default=None,
                        help="Comma-separated list of animal class names (overrides --animals-dir)")
    parser.add_argument("--train-data", type=str, default=None,
                        help="Optional JSONL file with training examples; each line should be a JSON object with 'text' and 'entities'")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of synthetic samples to generate when no training file is provided")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased",
                        help="Hugging Face model name for token classification")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "ner_model"),
        help="Directory to save the trained NER model"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max token length")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. cuda or cpu)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.animals:
        animals = [a.strip() for a in args.animals.split(",") if a.strip()]
    else:
        animals = load_animals_from_dir(args.animals_dir)

    if not animals:
        raise ValueError("No animal classes found. Provide --animals or point --animals-dir to a folder with subfolders.")

    if args.train_data:
        print(f"Loading training data from {args.train_data}")
        texts, entities = load_training_data_from_jsonl(args.train_data)
    else:
        print(f"Generating {args.num_samples} synthetic training samples for {len(animals)} animals")
        texts, entities = generate_synthetic_data(animals, num_samples=args.num_samples)

    ner = AnimalNER(model_name=args.model_name, max_length=args.max_length, device=args.device)
    train_data = ner.prepare_training_data(texts, entities)

    ner.train(
        train_data=train_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()
