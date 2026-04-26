# server/text_model.py

"""
Clinical Text Diagnosis Model (PyTorch)
=========================================
Text classification model for clinical notes / doctor's notes.

Use Case: Clinical notes → Disease prediction
Framework: PyTorch (torch.nn)
Architecture: Embedding + LSTM/Linear classifier

In federated learning:
  - Each hospital has its OWN patient clinical notes (EHR text)
  - Hospital trains text model locally on its notes
  - Only model WEIGHTS are sent to aggregator
  - NO patient notes/text are shared!

PyTorch Components Used:
  - torch.nn.Embedding       → Convert words to vectors
  - torch.nn.Linear          → Classification layers
  - torch.nn.ReLU            → Activation
  - torch.nn.Dropout         → Regularization
"""

import torch
import torch.nn as nn


class ClinicalTextModel(nn.Module):
    """
    Clinical Text Classification Model
    
    Architecture:
        Input: Bag-of-words vector (vocab_size features)
            ↓
        Linear(vocab_size → 256) + ReLU + Dropout
            ↓
        Linear(256 → 128) + ReLU + Dropout
            ↓
        Linear(128 → 64) + ReLU
            ↓
        Linear(64 → num_classes) → Output
    
    Input:
        Bag-of-words representation of clinical notes
        Each note is converted to a fixed-size vector
    
    Output:
        Disease category prediction:
          Class 0: Healthy / No significant findings
          Class 1: Cardiovascular disease
          Class 2: Respiratory disease
          Class 3: Neurological condition
          Class 4: Musculoskeletal disorder
    """

    def __init__(self, vocab_size=500, num_classes=5):
        super(ClinicalTextModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Linear(vocab_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, num_classes)
        )
        
        self.total_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """
        Forward pass: Text features → Prediction
        
        Args:
            x: Tensor of shape (batch_size, vocab_size) — bag-of-words vector
        
        Returns:
            Tensor of shape (batch_size, num_classes) — class probabilities
        """
        return self.classifier(x)
    
    def get_model_info(self):
        return {
            "type": "Text-Based Clinical Diagnosis",
            "architecture": "Multi-Layer Perceptron (Bag-of-Words)",
            "framework": "PyTorch",
            "input": f"Bag-of-words vector ({self.vocab_size} features)",
            "output": f"{self.num_classes} disease categories",
            "total_parameters": self.total_params,
            "classes": [
                "Healthy / No significant findings",
                "Cardiovascular disease",
                "Respiratory disease",
                "Neurological condition",
                "Musculoskeletal disorder"
            ],
            "use_case": "Clinical notes → Disease prediction",
            "data_source": "Hospital EHR clinical notes (simulated)"
        }


def create_text_model(vocab_size=500, num_classes=5):
    """Create a fresh text classification model"""
    return ClinicalTextModel(vocab_size=vocab_size, num_classes=num_classes)