"""
Unified Pipeline: Named Entity Recognition + Image Classification

This pipeline takes text and image inputs and determines if the text correctly
describes what animal is in the image.

Flow:
1. Extract animal entities from text using NER
2. Classify the animal in the image
3. Compare extracted entities with classification
4. Return boolean result (True if match, False otherwise)
"""

import os
import json
from typing import Dict, List, Tuple, Optional
import re
from pathlib import Path

import numpy as np
from PIL import Image

from ner.model import AnimalNER
from image_classifier.model import AnimalImageClassifier


class AnimalVerificationPipeline:
    """Pipeline for verifying if text matches image content."""
    
    def __init__(
        self,
        ner_model_path: Optional[str] = None,
        image_classifier_path: Optional[str] = None,
        animal_classes: Optional[List[str]] = None,
        ner_confidence_threshold: float = 0.5,
        image_confidence_threshold: float = 0.5,
        use_fuzzy_matching: bool = True
    ):
        """
        Initialize the pipeline.
        
        Args:
            ner_model_path: Path to trained NER model
            image_classifier_path: Path to trained image classifier
            animal_classes: List of animal class names
            ner_confidence_threshold: Confidence threshold for NER
            image_confidence_threshold: Confidence threshold for image classifier
            use_fuzzy_matching: Whether to use fuzzy string matching
        """
        self.ner_confidence_threshold = ner_confidence_threshold
        self.image_confidence_threshold = image_confidence_threshold
        self.use_fuzzy_matching = use_fuzzy_matching
        
        # Determine animal classes
        self.animal_classes = animal_classes or []

        # If no classes were provided, try to infer from the image classifier metadata
        def _load_classes_from_model(model_dir: Optional[str]) -> List[str]:
            if not model_dir:
                return []
            metadata_path = os.path.join(model_dir, "metadata.json")
            if not os.path.exists(metadata_path):
                return []
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                return metadata.get("class_names", []) or []
            except Exception:
                return []

        if not self.animal_classes:
            self.animal_classes = _load_classes_from_model(image_classifier_path)

        # Fallback default (Animals-10)
        if not self.animal_classes:
            self.animal_classes = [
                "cat", "dog", "horse", "elephant",
                "butterfly", "chicken", "cow", "sheep",
                "spider", "squirrel"
            ]

        # Initialize models
        print("Initializing NER model...")
        self.ner_model = AnimalNER()
        if ner_model_path and os.path.exists(ner_model_path):
            self.ner_model.load_model(ner_model_path)
            print(f"  Loaded from {ner_model_path}")
        else:
            print("  Using base model (not fine-tuned)")

        print("Initializing Image Classification model...")
        self.image_classifier = AnimalImageClassifier(
            num_classes=len(self.animal_classes),
            model_name="resnet50",
            pretrained=True
        )
        self.image_classifier.set_class_names(self.animal_classes)
        if image_classifier_path and os.path.exists(image_classifier_path):
            self.image_classifier.load_model(image_classifier_path)
            print(f"  Loaded from {image_classifier_path}")
        else:
            print("  Using pre-trained ResNet50 (not fine-tuned)")
    
    def extract_animals_from_text(self, text: str) -> List[str]:
        """
        Extract animal entities from text.
        
        Args:
            text: Input text
        
        Returns:
            List of extracted animal entities (lowercase)
        """
        # Use NER model
        ner_result = self.ner_model.predict(
            text,
            confidence_threshold=self.ner_confidence_threshold
        )
        extracted_entities = ner_result.get("entity_types", [])
        
        # Additional rule-based extraction for common patterns
        text_lower = text.lower()
        additional_animals = []
        for animal in self.animal_classes:
            if animal in text_lower:
                if animal not in extracted_entities:
                    additional_animals.append(animal)
        
        all_animals = extracted_entities + additional_animals
        return list(set([a.lower() for a in all_animals]))  # Remove duplicates
    
    def classify_image(self, image_path: str) -> Dict[str, any]:
        """
        Classify animal in image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Classification result with top prediction and confidence
        """
        result = self.image_classifier.predict(
            image_path,
            top_k=5,
            return_probabilities=True
        )
        
        top_prediction = result["top_prediction"]
        return {
            "class": top_prediction["class"].lower(),
            "confidence": top_prediction["confidence"],
            "top_5": result["top_k_predictions"]
        }
    
    def fuzzy_match(self, entity: str, prediction: str, threshold: float = 0.8) -> bool:
        """
        Simple fuzzy matching between entity and prediction.
        
        Args:
            entity: Extracted entity from text
            prediction: Predicted class from image
            threshold: Similarity threshold
        
        Returns:
            True if entities match (with threshold)
        """
        entity = entity.lower().strip()
        prediction = prediction.lower().strip()
        
        # Exact match
        if entity == prediction:
            return True
        
        # Substring match
        if entity in prediction or prediction in entity:
            return True
        
        # Common variations
        variations = {
            "cat": ["kitten", "feline", "kitty"],
            "dog": ["puppy", "canine", "doggy"],
            "bird": ["eagle", "parrot", "sparrow", "dove", "chicken"],
            "horse": ["pony", "stallion", "mare"],
            "monkey": ["primate", "ape"],
        }
        
        for base, aliases in variations.items():
            if (entity == base and prediction in aliases) or \
               (prediction == base and entity in aliases):
                return True
        
        return False
    
    def verify(
        self,
        text: str,
        image_path: str,
        return_details: bool = False
    ) -> Dict[str, any]:
        """
        Verify if text matches image content.
        
        Args:
            text: Input text describing an animal
            image_path: Path to image
            return_details: Whether to return detailed analysis
        
        Returns:
            Dictionary with verification result
        """
        # Validate inputs
        if not text or not text.strip():
            return {
                "match": False,
                "error": "Empty text input",
                "details": None if not return_details else {}
            }
        
        if not os.path.exists(image_path):
            return {
                "match": False,
                "error": f"Image not found: {image_path}",
                "details": None if not return_details else {}
            }
        
        try:
            Image.open(image_path)
        except Exception as e:
            return {
                "match": False,
                "error": f"Invalid image: {str(e)}",
                "details": None if not return_details else {}
            }
        
        # Extract animals from text
        extracted_animals = self.extract_animals_from_text(text)
        
        # Classify image
        image_classification = self.classify_image(image_path)
        predicted_animal = image_classification["class"]
        confidence = image_classification["confidence"]
        
        # Check if no animals were extracted
        if not extracted_animals:
            match = False
            reason = "No animal entities found in text"
        # Check if confidence is too low
        elif confidence < self.image_confidence_threshold:
            match = False
            reason = f"Image classification confidence ({confidence:.2f}) below threshold"
        else:
            # Try to match extracted animals with predicted animal
            match = False
            for entity in extracted_animals:
                if self.use_fuzzy_matching:
                    if self.fuzzy_match(entity, predicted_animal):
                        match = True
                        break
                else:
                    if entity.lower() == predicted_animal.lower():
                        match = True
                        break
            
            reason = "Match found" if match else "No matching animal in text"
        
        result = {
            "match": match,
            "confidence": confidence,
            "reason": reason
        }
        
        if return_details:
            result["details"] = {
                "text": text,
                "image_path": image_path,
                "extracted_animals": extracted_animals,
                "predicted_animal": predicted_animal,
                "image_confidence": confidence,
                "top_5_predictions": image_classification.get("top_5", [])
            }
        
        return result
    
    def batch_verify(
        self,
        text_image_pairs: List[Tuple[str, str]],
        return_details: bool = False
    ) -> List[Dict]:
        """
        Verify multiple text-image pairs.
        
        Args:
            text_image_pairs: List of (text, image_path) tuples
            return_details: Whether to return detailed analysis
        
        Returns:
            List of verification results
        """
        results = []
        for text, image_path in text_image_pairs:
            result = self.verify(text, image_path, return_details=return_details)
            results.append(result)
        
        return results
    
    def save_config(self, config_path: str):
        """Save pipeline configuration."""
        config = {
            "animal_classes": self.animal_classes,
            "ner_confidence_threshold": self.ner_confidence_threshold,
            "image_confidence_threshold": self.image_confidence_threshold,
            "use_fuzzy_matching": self.use_fuzzy_matching
        }
        os.makedirs(os.path.dirname(config_path) or ".", exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    
    def load_config(self, config_path: str):
        """Load pipeline configuration."""
        with open(config_path, "r") as f:
            config = json.load(f)
        
        self.animal_classes = config.get("animal_classes", self.animal_classes)
        self.ner_confidence_threshold = config.get("ner_confidence_threshold", 0.5)
        self.image_confidence_threshold = config.get("image_confidence_threshold", 0.5)
        self.use_fuzzy_matching = config.get("use_fuzzy_matching", True)


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the animal verification pipeline (text + image -> boolean match)"
    )
    parser.add_argument("--text", type=str, required=True, help="Input text describing an animal")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--ner-model", type=str, default=None, help="Path to fine-tuned NER model directory")
    parser.add_argument("--image-model", type=str, default=None, help="Path to trained image classifier model directory")
    parser.add_argument("--ner-threshold", type=float, default=0.5, help="NER confidence threshold")
    parser.add_argument("--image-threshold", type=float, default=0.5, help="Image classification confidence threshold")
    parser.add_argument("--no-fuzzy", action="store_true", help="Disable fuzzy matching")
    parser.add_argument("--print-details", action="store_true", help="Print detailed pipeline results")
    parser.add_argument("--animal-classes", type=str, default=None,
                        help="Comma-separated list of animal classes (overrides model metadata)")

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    animal_classes = None
    if args.animal_classes:
        animal_classes = [c.strip() for c in args.animal_classes.split(",") if c.strip()]

    pipeline = AnimalVerificationPipeline(
        ner_model_path=args.ner_model,
        image_classifier_path=args.image_model,
        animal_classes=animal_classes,
        ner_confidence_threshold=args.ner_threshold,
        image_confidence_threshold=args.image_threshold,
        use_fuzzy_matching=not args.no_fuzzy
    )

    result = pipeline.verify(args.text, args.image, return_details=args.print_details)

    print("MATCH:" if result["match"] else "NO MATCH")
    if args.print_details:
        print(json.dumps(result, indent=2))
