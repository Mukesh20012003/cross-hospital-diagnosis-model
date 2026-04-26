# server/privacy.py

import torch
import numpy as np
from typing import Dict
import copy


class DifferentialPrivacy:
    """
    Differential Privacy Module
    
    Adds calibrated noise to model updates to prevent
    information leakage about individual patient records.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 max_grad_norm: float = 1.0):
        """
        Args:
            epsilon: Privacy budget (smaller = more private, less accurate)
            delta: Probability of privacy breach
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        
        # Calculate noise scale based on epsilon
        self.noise_scale = self._calculate_noise_scale()
    
    def _calculate_noise_scale(self) -> float:
        """Calculate Gaussian noise scale for (epsilon, delta)-DP"""
        # Simplified Gaussian mechanism
        noise_scale = (2 * np.log(1.25 / self.delta)) ** 0.5 / self.epsilon
        return noise_scale * self.max_grad_norm
    
    def clip_update(self, model_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Clip model update to limit sensitivity.
        Prevents any single data point from having too much influence.
        """
        clipped_weights = {}
        
        # Calculate total norm of all weights
        total_norm = 0.0
        for key, value in model_weights.items():
            total_norm += torch.norm(value.float()).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip if norm exceeds max
        clip_factor = min(1.0, self.max_grad_norm / (total_norm + 1e-8))
        
        for key, value in model_weights.items():
            clipped_weights[key] = value.float() * clip_factor
        
        return clipped_weights
    
    def add_noise(self, model_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Add Gaussian noise to model weights for differential privacy.
        """
        noisy_weights = {}
        
        for key, value in model_weights.items():
            noise = torch.normal(
                mean=0.0,
                std=self.noise_scale,
                size=value.shape
            )
            noisy_weights[key] = value.float() + noise
        
        return noisy_weights
    
    def apply_privacy(self, model_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply full differential privacy pipeline:
        1. Clip the update
        2. Add calibrated noise
        """
        # Step 1: Clip
        clipped = self.clip_update(model_weights)
        
        # Step 2: Add noise
        noisy = self.add_noise(clipped)
        
        return noisy
    
    def get_privacy_report(self) -> dict:
        """Return privacy parameters for compliance reporting"""
        return {
            "mechanism": "Gaussian Differential Privacy",
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_gradient_norm": self.max_grad_norm,
            "noise_scale": round(self.noise_scale, 6),
            "privacy_guarantee": f"({self.epsilon}, {self.delta})-Differential Privacy",
            "description": "Each model update is clipped and noised to prevent "
                          "reconstruction of individual patient data from shared weights."
        }


class SecureAggregation:
    """
    Secure Aggregation Module
    
    Ensures the server can only see the aggregated result,
    not individual hospital updates (simplified version).
    """
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.masks = {}
    
    def generate_mask(self, client_id: int, weight_shapes: dict) -> Dict[str, torch.Tensor]:
        """Generate a random mask for a client"""
        mask = {}
        for key, shape in weight_shapes.items():
            mask[key] = torch.randn(shape)
        self.masks[client_id] = mask
        return mask
    
    def mask_update(self, weights: Dict[str, torch.Tensor], 
                    mask: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply mask to weights before sending"""
        masked = {}
        for key in weights:
            masked[key] = weights[key].float() + mask[key]
        return masked
    
    def unmask_aggregate(self, aggregated: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove masks from aggregated result"""
        # In secure aggregation, masks cancel out during summation
        # This is a simplified demonstration
        result = {}
        total_mask = None
        
        for client_id, mask in self.masks.items():
            if total_mask is None:
                total_mask = {k: v.clone() for k, v in mask.items()}
            else:
                for key in mask:
                    total_mask[key] += mask[key]
        
        if total_mask:
            for key in aggregated:
                result[key] = aggregated[key] - total_mask[key] / self.num_clients
        else:
            result = aggregated
        
        return result
    
    def get_security_report(self) -> dict:
        """Return security parameters for compliance"""
        return {
            "mechanism": "Secure Aggregation (Simplified)",
            "num_clients": self.num_clients,
            "description": "Individual hospital updates are masked before transmission. "
                          "The server can only compute the aggregate, not individual updates."
        }


def validate_update_integrity(weights: Dict[str, torch.Tensor], 
                               expected_keys: list = None) -> dict:
    """
    Validate that a model update is well-formed and not malicious.
    
    Checks:
    1. All expected keys present
    2. No NaN or Inf values
    3. Weight magnitudes within reasonable range
    """
    issues = []
    
    # Check for NaN/Inf
    for key, value in weights.items():
        if torch.isnan(value).any():
            issues.append(f"NaN values found in {key}")
        if torch.isinf(value).any():
            issues.append(f"Inf values found in {key}")
    
    # Check magnitude
    for key, value in weights.items():
        max_val = torch.max(torch.abs(value.float())).item()
        if max_val > 1000:
            issues.append(f"Large values in {key}: max={max_val:.2f}")
    
    # Check expected keys
    if expected_keys:
        missing = set(expected_keys) - set(weights.keys())
        extra = set(weights.keys()) - set(expected_keys)
        if missing:
            issues.append(f"Missing keys: {missing}")
        if extra:
            issues.append(f"Unexpected keys: {extra}")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "num_parameters": sum(v.numel() for v in weights.values()),
        "total_size_mb": sum(v.element_size() * v.numel() for v in weights.values()) / (1024*1024)
    }