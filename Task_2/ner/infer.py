"""
Inference script for the animal NER model.

This script loads a trained NER model and extracts animal entities from input text.
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so imports work from subfolders
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from ner.model import AnimalNER


def parse_args():
    parser = argparse.ArgumentParser(description="Run NER inference for animal extraction")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing the trained NER model")
    parser.add_argument("--text", type=str, default=None,
                        help="Single text input to run inference on")
    parser.add_argument("--input-file", type=str, default=None,
                        help="Path to a text file (one sample per line) to run inference on")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                        help="Minimum confidence threshold for extracted entities")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Optional path to write inference results as JSON")
    return parser.parse_args()


def main():
    args = parse_args()

    ner = AnimalNER()
    ner.load_model(args.model_dir)

    results = []

    if args.text:
        res = ner.predict(args.text, confidence_threshold=args.confidence_threshold)
        results.append(res)
        print(json.dumps(res, indent=2))

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                res = ner.predict(text, confidence_threshold=args.confidence_threshold)
                results.append(res)
                print(json.dumps(res, indent=2))

    if args.output_json and results:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
