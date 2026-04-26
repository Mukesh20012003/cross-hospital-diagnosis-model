# client/text_client.py

"""
Hospital Client for Clinical Text Federated Learning
======================================================
Trains a text classification model on local clinical notes
and submits weights to the aggregator server.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from server.text_model import ClinicalTextModel, create_text_model


class TextHospitalClient:
    """Hospital client for clinical text federated learning"""

    def __init__(self, hospital_name, server_url="http://127.0.0.1:8000"):
        self.hospital_name = hospital_name
        self.server_url = server_url
        self.text_dir = os.path.join("data", hospital_name, "text")
        self.model = None
        self.epochs = 10
        self.batch_size = 16
        self.learning_rate = 0.001

    def load_text_data(self):
        """Load clinical text data from local storage"""
        print(f"\n  📂 Loading text data for {self.hospital_name}...")
        
        data_path = os.path.join(self.text_dir, "text_data.pt")
        if not os.path.exists(data_path):
            print(f"  ❌ Text data not found! Run 'python -m client.prepare_text_data' first.")
            return None
        
        data = torch.load(data_path, weights_only=True)
        print(f"  ✅ Loaded: {len(data['X_train'])} train, {len(data['X_val'])} val notes")
        print(f"     Feature size: {data['X_train'].shape[1]} (vocabulary)")
        return data

    def train_locally(self, data):
        """Train text model on local clinical notes"""
        print(f"\n  🏋️ Training text model on {self.hospital_name}'s clinical notes...")
        
        vocab_size = data["X_train"].shape[1]
        
        if self.model is None:
            self.model = create_text_model(vocab_size=vocab_size, num_classes=5)
        
        train_dataset = TensorDataset(data["X_train"], data["y_train"])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            acc = correct / total
            avg_loss = total_loss / len(train_loader)
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f"     Epoch {epoch+1}/{self.epochs} — Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        
        # Validation
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(data["X_val"])
            val_loss = criterion(val_outputs, data["y_val"]).item()
            _, val_pred = torch.max(val_outputs, 1)
            val_acc = (val_pred == data["y_val"]).sum().item() / len(data["y_val"])
        
        print(f"\n  📊 Validation — Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        return round(val_acc, 4), round(val_loss, 4)

    def save_weights(self):
        """Save model weights locally"""
        weights_path = os.path.join(self.text_dir, "text_weights.pt")
        torch.save(self.model.state_dict(), weights_path)
        return weights_path

    def run_training(self):
        """Complete local training pipeline"""
        print(f"\n  {'=' * 50}")
        print(f"  📝 {self.hospital_name} — Text Diagnosis Training")
        print(f"  {'=' * 50}")
        
        data = self.load_text_data()
        if data is None:
            return None, None
        
        accuracy, loss = self.train_locally(data)
        weights_path = self.save_weights()
        
        print(f"\n  ✅ {self.hospital_name} text training complete!")
        print(f"     Accuracy: {accuracy:.4f}")
        print(f"     Weights saved: {weights_path}")
        
        return accuracy, loss


def run_text_demo():
    """Demo: Train text model across 3 hospitals"""
    print("\n" + "=" * 60)
    print("  📝 CLINICAL TEXT FEDERATED LEARNING DEMO")
    print("  Model: MLP for Clinical Notes Classification")
    print("  Classes: Healthy | Cardiovascular | Respiratory |")
    print("           Neurological | Musculoskeletal")
    print("=" * 60)
    
    results = {}
    
    for i in range(1, 4):
        client = TextHospitalClient(f"hospital_{i}")
        acc, loss = client.run_training()
        if acc is not None:
            results[f"hospital_{i}"] = {"accuracy": acc, "loss": loss}
    
    # Summary
    print("\n" + "=" * 60)
    print("  📊 TEXT TRAINING RESULTS")
    print("=" * 60)
    for name, res in results.items():
        print(f"  {name}: Accuracy={res['accuracy']:.4f}, Loss={res['loss']:.4f}")
    
    if results:
        avg_acc = sum(r["accuracy"] for r in results.values()) / len(results)
        print(f"\n  Average Accuracy: {avg_acc:.4f}")
        print(f"  🔒 No clinical notes were shared — only model weights!")


if __name__ == "__main__":
    # Check if text data exists
    if not os.path.exists("data/hospital_1/text"):
        print("⚠️ Text data not found! Preparing first...")
        from client.prepare_text_data import prepare_text_dataset
        prepare_text_dataset()
    
    run_text_demo()