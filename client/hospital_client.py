# client/hospital_client.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import requests
import json
import time
from datetime import datetime

from server.model_definition import DiagnosisModel, create_initial_model


class HospitalClient:
    """
    Hospital Node Client for Federated Learning
    
    Each hospital:
    1. Downloads the global model from the server
    2. Trains it on local data
    3. Sends updated weights back to the server
    4. Never shares raw patient data!
    """

    def __init__(self, hospital_name: str, server_url: str = "http://127.0.0.1:8000"):
        self.hospital_name = hospital_name
        self.server_url = server_url
        self.api_key = None
        self.data_dir = os.path.join("data", hospital_name)
        self.model = None
        
        # Training hyperparameters
        self.epochs = 5
        self.batch_size = 32
        self.learning_rate = 0.001
        
        print(f"\n{'='*50}")
        print(f"  🏥 Hospital Client: {hospital_name}")
        print(f"  📡 Server: {server_url}")
        print(f"{'='*50}")

    def register(self):
        """Register this hospital with the central server"""
        print(f"\n📝 Registering {self.hospital_name} with server...")
        
        # Load metadata to get data size
        data_size = 0
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            # Handle different metadata formats
            if "total_samples" in metadata:
                data_size = metadata["total_samples"]
            elif "total_records" in metadata:
                data_size = metadata["total_records"]
            elif "train_samples" in metadata:
                data_size = metadata["train_samples"]
            else:
                data_size = 300  # Default
        else:
            # Try to get size from data.pt file
            data_path = os.path.join(self.data_dir, "data.pt")
            if os.path.exists(data_path):
                data = torch.load(data_path, weights_only=True)
                data_size = len(data["X_train"]) + len(data["X_val"])
            else:
                data_size = 300  # Default
        
        payload = {
            "name": self.hospital_name,
            "location": f"Location_{self.hospital_name}",
            "data_size": data_size
        }
        
        try:
            response = requests.post(f"{self.server_url}/register_hospital", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                self.api_key = result["api_key"]
                print(f"   ✅ Registered! API Key: {self.api_key[:20]}...")
                
                # Save API key locally
                key_file = os.path.join(self.data_dir, "api_key.txt")
                with open(key_file, "w") as f:
                    f.write(self.api_key)
                
                return True
            elif response.status_code == 400:
                # Already registered, load existing key
                print(f"   ⚠️ Already registered. Loading existing API key...")
                key_file = os.path.join(self.data_dir, "api_key.txt")
                if os.path.exists(key_file):
                    with open(key_file, "r") as f:
                        self.api_key = f.read().strip()
                    print(f"   ✅ Loaded API Key: {self.api_key[:20]}...")
                    return True
                else:
                    print(f"   ❌ No saved API key found. Please re-register with a new name.")
                    return False
            else:
                print(f"   ❌ Registration failed: {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Cannot connect to server at {self.server_url}")
            print(f"   Make sure the server is running!")
            return False

    def load_local_data(self):
        """Load this hospital's local dataset"""
        print(f"\n📂 Loading local data for {self.hospital_name}...")
        
        data_path = os.path.join(self.data_dir, "data.pt")
        if not os.path.exists(data_path):
            print(f"   ❌ Data file not found: {data_path}")
            print(f"   Run 'python -m client.prepare_data' first!")
            return None
        
        data = torch.load(data_path, weights_only=True)
        
        print(f"   ✅ Training samples: {len(data['X_train'])}")
        print(f"   ✅ Validation samples: {len(data['X_val'])}")
        
        return data

    def download_global_model(self):
        """Download the latest global model from the server"""
        print(f"\n📥 Downloading global model from server...")
        
        try:
            params = {"api_key": self.api_key} if self.api_key else {}
            response = requests.get(f"{self.server_url}/get_global_model", params=params)
            
            if response.status_code == 200:
                # Save model temporarily
                temp_path = os.path.join(self.data_dir, "global_model_temp.pt")
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                
                # Load into model
                self.model = create_initial_model()
                weights = torch.load(temp_path, map_location="cpu", weights_only=True)
                self.model.load_state_dict(weights)
                
                print(f"   ✅ Global model downloaded and loaded!")
                return True
            else:
                print(f"   ❌ Failed to download: {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Cannot connect to server!")
            return False

    def train_locally(self, data):
        """
        Train the model on LOCAL data only.
        No patient data leaves this function!
        """
        print(f"\n🏋️ Training locally on {self.hospital_name}'s data...")
        print(f"   Epochs: {self.epochs}, Batch size: {self.batch_size}, LR: {self.learning_rate}")
        
        if self.model is None:
            self.model = create_initial_model()
        
        # Create data loaders
        train_dataset = TensorDataset(data["X_train"], data["y_train"])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        val_dataset = TensorDataset(data["X_val"], data["y_val"])
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
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
            
            train_acc = correct / total
            avg_loss = total_loss / len(train_loader)
            
            if (epoch + 1) % 1 == 0:
                print(f"   Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        # Validation
        val_accuracy, val_loss = self.evaluate(data["X_val"], data["y_val"])
        print(f"\n   📊 Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        return val_accuracy, val_loss

    def evaluate(self, X, y):
        """Evaluate model on given data"""
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            outputs = self.model(X)
            loss = criterion(outputs, y)
            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == y).sum().item() / y.size(0)
        
        self.model.train()
        return round(accuracy, 4), round(loss.item(), 4)

    def submit_weights(self, data_size: int, accuracy: float, loss: float):
        """
        Submit trained model weights to the aggregator server.
        ONLY weights are sent — NO raw data!
        """
        print(f"\n📤 Submitting weights to server...")
        
        # Save weights to file
        weights_path = os.path.join(self.data_dir, "local_weights.pt")
        torch.save(self.model.state_dict(), weights_path)
        
        try:
            with open(weights_path, "rb") as f:
                files = {"weights_file": ("weights.pt", f, "application/octet-stream")}
                form_data = {
                    "api_key": self.api_key,
                    "data_size": str(data_size),
                    "local_accuracy": str(accuracy),
                    "local_loss": str(loss)
                }
                
                response = requests.post(
                    f"{self.server_url}/submit_update",
                    data=form_data,
                    files=files
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Weights submitted successfully!")
                print(f"   📊 Updates received: {result['updates_received']}/{result['target']}")
                return True
            else:
                print(f"   ❌ Submission failed: {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Cannot connect to server!")
            return False

    def run_one_round(self):
        """
        Complete one round of federated learning:
        1. Download global model
        2. Train locally
        3. Submit updated weights
        """
        print(f"\n{'='*50}")
        print(f"  🔄 {self.hospital_name} - Starting Training Round")
        print(f"{'='*50}")
        
        # Step 1: Load local data
        data = self.load_local_data()
        if data is None:
            return False
        
        # Step 2: Download global model
        if not self.download_global_model():
            return False
        
        # Step 3: Train locally
        accuracy, loss = self.train_locally(data)
        
        # Step 4: Submit weights
        data_size = len(data["X_train"])
        success = self.submit_weights(data_size, accuracy, loss)
        
        if success:
            print(f"\n   🎉 {self.hospital_name} completed this round!")
            print(f"   📊 Local Accuracy: {accuracy:.4f}")
        
        return success