# server/image_model.py

"""
Medical Image Diagnosis Model (PyTorch CNN)
=============================================
Convolutional Neural Network for medical image classification.

Use Case: Chest X-ray → Normal / Pneumonia / COVID-19
Framework: PyTorch (torch.nn)
Architecture: CNN with Conv2d, MaxPool2d, Fully Connected layers

In federated learning:
  - Each hospital has its OWN chest X-ray images (DICOM/PACS)
  - Hospital trains CNN locally on its images
  - Only CNN WEIGHTS are sent to aggregator
  - NO patient images are shared!

PyTorch Components Used:
  - torch.nn.Conv2d       → Convolutional layers (extract image features)
  - torch.nn.MaxPool2d    → Pooling layers (reduce dimensions)
  - torch.nn.BatchNorm2d  → Normalize feature maps
  - torch.nn.Linear       → Fully connected classification layers
  - torch.nn.ReLU         → Activation function
  - torch.nn.Dropout      → Regularization
"""

import torch
import torch.nn as nn


class MedicalImageModel(nn.Module):
    """
    CNN for Medical Image Classification
    
    Architecture:
        Input (1 x 64 x 64 grayscale medical image)
            ↓
        Conv2d(1→32) + ReLU + BatchNorm + MaxPool
            ↓
        Conv2d(32→64) + ReLU + BatchNorm + MaxPool
            ↓
        Conv2d(64→128) + ReLU + BatchNorm + MaxPool
            ↓
        Flatten → Linear(128*8*8 → 256) + ReLU + Dropout
            ↓
        Linear(256 → 3) → Output (Normal / Pneumonia / COVID)
    
    Input: 
        Grayscale medical image (1 channel, 64x64 pixels)
        In real hospital: DICOM images from PACS system
    
    Output:
        3 classes:
          Class 0: Normal (healthy lungs)
          Class 1: Pneumonia
          Class 2: COVID-19
    """

    def __init__(self, num_classes=3):
        super(MedicalImageModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # Block 1: 64x64 → 32x32
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            
            # Block 2: 32x32 → 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            
            # Block 3: 16x16 → 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        self.total_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        """
        Forward pass: Image → Prediction
        
        Args:
            x: Tensor of shape (batch_size, 1, 64, 64) — grayscale image
        
        Returns:
            Tensor of shape (batch_size, 3) — class probabilities
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def get_model_info(self):
        return {
            "type": "Medical Image Diagnosis",
            "architecture": "Convolutional Neural Network (CNN)",
            "framework": "PyTorch",
            "input": "Grayscale image (1 x 64 x 64)",
            "output": f"{self.num_classes} classes (Normal / Pneumonia / COVID-19)",
            "total_parameters": self.total_params,
            "layers": [
                "Conv2d(1→32) + ReLU + BatchNorm + MaxPool",
                "Conv2d(32→64) + ReLU + BatchNorm + MaxPool",
                "Conv2d(64→128) + ReLU + BatchNorm + MaxPool",
                "Flatten → Linear(8192→256) + ReLU + Dropout(0.5)",
                "Linear(256→3) → Output"
            ],
            "use_case": "Chest X-ray classification",
            "data_source": "Hospital PACS/DICOM system (simulated)"
        }


def create_image_model(num_classes=3):
    """Create a fresh image classification model"""
    return MedicalImageModel(num_classes=num_classes)