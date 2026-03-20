"""
Inference script for the animal image classification model.

Usage:
  python infer_image_classifier.py --model-dir ./models/image_classifier --image path/to/image.jpg
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work from subfolders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from image_classifier.model import AnimalImageClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Run image classification inference")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing the trained image classifier model")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top predictions to return")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional path to write inference result as JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    classifier = AnimalImageClassifier()
    classifier.load_model(args.model_dir)

    result = classifier.predict(args.image, top_k=args.top_k, return_probabilities=True)
    print(json.dumps(result, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
