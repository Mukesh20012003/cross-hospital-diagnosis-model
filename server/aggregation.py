# server/aggregation.py

import torch
import os
import copy
from typing import List, Dict
from server.model_definition import DiagnosisModel, create_initial_model


def federated_averaging(model_paths: List[str], data_sizes: List[int]) -> dict:
    """
    Federated Averaging (FedAvg) Algorithm
    
    Combines model weights from multiple hospitals.
    Each hospital's weights are weighted by its data size.
    
    Formula: global_weights = SUM(data_size_i * weights_i) / SUM(data_sizes)
    
    Args:
        model_paths: List of file paths to hospital model weights
        data_sizes: List of data sizes for each hospital
    
    Returns:
        Aggregated global model state_dict
    """
    
    if len(model_paths) == 0:
        raise ValueError("No model updates to aggregate!")
    
    if len(model_paths) != len(data_sizes):
        raise ValueError("Number of models and data sizes must match!")
    
    # Total data points across all hospitals
    total_data = sum(data_sizes)
    
    if total_data == 0:
        raise ValueError("Total data size cannot be zero!")
    
    # Load first model to get structure
    first_weights = torch.load(model_paths[0], map_location="cpu", weights_only=True)
    
    # Initialize averaged weights with zeros
    averaged_weights = {}
    for key in first_weights.keys():
        averaged_weights[key] = torch.zeros_like(first_weights[key], dtype=torch.float32)
    
    # Weighted average of all models
    for i, model_path in enumerate(model_paths):
        model_weights = torch.load(model_path, map_location="cpu", weights_only=True)
        weight_factor = data_sizes[i] / total_data  # Proportional to data size
        
        for key in averaged_weights.keys():
            averaged_weights[key] += model_weights[key].float() * weight_factor
    
    return averaged_weights


def aggregate_and_save(model_paths: List[str], data_sizes: List[int], 
                       save_path: str) -> dict:
    """
    Perform FedAvg and save the global model
    
    Args:
        model_paths: Paths to hospital weight files
        data_sizes: Data sizes for each hospital
        save_path: Where to save the aggregated model
    
    Returns:
        Aggregated weights dictionary
    """
    
    # Perform federated averaging
    global_weights = federated_averaging(model_paths, data_sizes)
    
    # Save global model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(global_weights, save_path)
    
    print(f"✅ Global model saved to: {save_path}")
    return global_weights


def evaluate_global_model(model_weights: dict, test_data, test_labels) -> dict:
    """
    Evaluate the global model on a test dataset
    
    Args:
        model_weights: Global model state_dict
        test_data: Test features tensor
        test_labels: Test labels tensor
    
    Returns:
        Dictionary with accuracy and loss
    """
    
    model = create_initial_model()
    model.load_state_dict(model_weights)
    model.eval()
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        outputs = model(test_data)
        loss = criterion(outputs, test_labels)
        
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == test_labels).sum().item()
        total = test_labels.size(0)
        accuracy = correct / total
    
    return {
        "accuracy": round(accuracy, 4),
        "loss": round(loss.item(), 4)
    }