"""
Named Entity Recognition (NER) model for extracting animal entities from text.
Uses transformer-based models from Hugging Face.

This module provides:
- NER model training with fine-tuning on animal entity extraction
- Inference for extracting animal mentions from text
- Support for model saving and loading
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pickle

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)


class AnimalNER:
    """Named Entity Recognition model for animal entity extraction."""
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 3,  # O, B-ANIMAL, I-ANIMAL
        max_length: int = 128,
        device: str = None
    ):
        """
        Initialize NER model.
        
        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of entity labels (3: O, B-ANIMAL, I-ANIMAL)
            max_length: Maximum sequence length
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if device is None:
            self.device = "cuda" if cuda_available else "cpu"
        else:
            self.device = device

        self.label2id = {"O": 0, "B-ANIMAL": 1, "I-ANIMAL": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            num_labels=num_labels
        ).to(self.device)
    
    def prepare_training_data(
        self,
        texts: List[str],
        animal_entities: List[List[str]],
        save_path: Optional[str] = None
    ) -> List[Dict]:
        """
        Prepare training data with IOB tagging.
        
        Args:
            texts: List of text samples
            animal_entities: List of lists containing animal entities per text
            save_path: Optional path to save processed data
        
        Returns:
            List of processed training samples
        """
        training_data = []
        
        for text, entities in zip(texts, animal_entities):
            # Tokenize text
            encoding = self.tokenizer(
                text.split(),
                truncation=True,
                max_length=self.max_length,
                is_split_into_words=True
            )
            
            # Create labels (IOB format)
            labels = [0] * len(encoding["input_ids"])  # Start with O labels
            
            # Simple label assignment based on entity presence
            word_ids = encoding.word_ids()
            for idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    word = text.split()[word_id]
                    for entity in entities:
                        if word.lower() == entity.lower():
                            labels[idx] = self.label2id.get("B-ANIMAL", 1)
            
            encoding["labels"] = labels
            training_data.append(encoding)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "wb") as f:
                pickle.dump(training_data, f)
        
        return training_data
    
    def train(
        self,
        train_data: List[Dict],
        output_dir: str = "./models/ner_model",
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01
    ):
        """
        Train the NER model.
        
        Args:
            train_data: Prepared training data
            output_dir: Directory to save trained model
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dataset class
        class NERDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                item = self.data[idx]
                return {
                    "input_ids": torch.tensor(item["input_ids"]),
                    "attention_mask": torch.tensor(item["attention_mask"]),
                    "labels": torch.tensor(item["labels"])
                }
        
        dataset = NERDataset(train_data)
        data_collator = DataCollatorForTokenClassification(self.tokenizer)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            save_steps=100,
            save_total_limit=2,
            logging_steps=10
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator
        )
        
        trainer.train()
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    def predict(
        self,
        text: str,
        confidence_threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Extract animal entities from text.
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence for entity extraction
        
        Returns:
            Dictionary with extracted entities and their positions
        """
        self.model.eval()
        
        encoding = self.tokenizer(
            text.split(),
            truncation=True,
            max_length=self.max_length,
            is_split_into_words=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
        
        # Get predictions
        predictions = torch.argmax(logits, dim=2)[0].cpu().numpy()
        probabilities = torch.softmax(logits, dim=2)[0].cpu().numpy()
        
        # Extract entities
        entities = []
        word_ids = encoding.word_ids()
        words = text.split()
        
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if pred != 0:  # Not O label
                word_id = word_ids[idx]
                if word_id is not None and word_id < len(words):
                    max_prob = prob[pred]
                    if max_prob >= confidence_threshold:
                        label = self.id2label[pred]
                        entities.append({
                            "word": words[word_id],
                            "label": label,
                            "confidence": float(max_prob)
                        })
        
        # Remove duplicates and consolidate
        unique_entities = {}
        for entity in entities:
            word_lower = entity["word"].lower()
            if word_lower not in unique_entities:
                unique_entities[word_lower] = entity
        
        return {
            "text": text,
            "entities": list(unique_entities.values()),
            "entity_types": [e["word"].lower() for e in unique_entities.values()]
        }
    
    def load_model(self, model_path: str):
        """Load pre-trained model."""
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Model loaded from {model_path}")
