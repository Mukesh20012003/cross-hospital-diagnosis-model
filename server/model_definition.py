# server/model_definition.py

import torch
import torch.nn as nn


class DiagnosisModel(nn.Module):
    """
    Simple neural network for diagnosis prediction.
    Works with tabular health data (e.g., patient vitals, lab results).
    
    Input: 13 features (like heart disease dataset)
    Output: 2 classes (disease / no disease)
    """

    def __init__(self, input_dim=13, hidden_dim=64, output_dim=2):
        super(DiagnosisModel, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            
            # Layer 2
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(0.3),
            
            # Layer 3
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            
            # Output Layer
            nn.Linear(hidden_dim // 4, output_dim)
        )

    def forward(self, x):
        return self.network(x)


def create_initial_model(input_dim=13, hidden_dim=64, output_dim=2):
    """Create and return a fresh model"""
    model = DiagnosisModel(input_dim, hidden_dim, output_dim)
    return model


def get_model_weights(model):
    """Extract model weights as a dictionary"""
    return {key: value.clone() for key, value in model.state_dict().items()}


def set_model_weights(model, weights):
    """Load weights into a model"""
    model.load_state_dict(weights)
    return model