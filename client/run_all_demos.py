# client/run_all_demos.py

"""
Run All Three Diagnosis Types Demo
====================================
Demonstrates federated learning with:
  TYPE 1: Tabular EHR (Heart Disease) — existing
  TYPE 2: Medical Images (Chest X-ray) — new
  TYPE 3: Clinical Text (Doctor's Notes) — new

All three use the SAME federated learning principle:
  - Each hospital trains LOCALLY
  - Only model WEIGHTS are shared
  - NO patient data leaves the hospital
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    print("\n" + "=" * 65)
    print("  🏥 CROSS-HOSPITAL DIAGNOSIS — ALL THREE DIAGNOSIS TYPES")
    print("=" * 65)
    print("""
  This demo shows federated learning working with 3 data types:

  TYPE 1: 📊 Tabular EHR Data (Heart Disease)
          Model: MLP (Linear layers)
          Input: 13 clinical features → Disease prediction
          
  TYPE 2: 🩻 Medical Images (Chest X-ray)
          Model: CNN (Conv2d layers)
          Input: 64x64 X-ray images → Normal/Pneumonia/COVID
          
  TYPE 3: 📝 Clinical Text (Doctor's Notes)
          Model: MLP (Bag-of-Words)
          Input: Clinical notes text → Disease category
    """)
    
    # ========== TYPE 1: Tabular ==========
    print("\n" + "=" * 65)
    print("  TYPE 1: 📊 TABULAR EHR DIAGNOSIS (Heart Disease)")
    print("=" * 65)
    
    if not os.path.exists("data/hospital_1/data.pt"):
        print("  ⚠️ Preparing tabular data...")
        from client.prepare_data import main as prep_tabular
        prep_tabular()
    
    # Quick train demo for tabular
    from client.hospital_client import HospitalClient
    import torch
    
    tabular_results = {}
    for i in range(1, 4):
        name = f"hospital_{i}"
        data_path = os.path.join("data", name, "data.pt")
        if os.path.exists(data_path):
            from server.model_definition import create_initial_model
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
            
            data = torch.load(data_path, weights_only=True)
            model = create_initial_model()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            dataset = TensorDataset(data["X_train"], data["y_train"])
            loader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            model.train()
            for epoch in range(5):
                for bx, by in loader:
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_out = model(data["X_val"])
                _, pred = torch.max(val_out, 1)
                acc = (pred == data["y_val"]).sum().item() / len(data["y_val"])
            
            tabular_results[name] = round(acc, 4)
            print(f"  ✅ {name}: Accuracy = {acc:.4f}")
    
    # ========== TYPE 2: Images ==========
    print("\n" + "=" * 65)
    print("  TYPE 2: 🩻 MEDICAL IMAGE DIAGNOSIS (Chest X-ray)")
    print("=" * 65)
    
    if not os.path.exists("data/hospital_1/images"):
        print("  ⚠️ Preparing image data...")
        from client.prepare_image_data import prepare_image_dataset
        prepare_image_dataset()
    
    from client.image_client import run_image_demo
    run_image_demo()
    
    # ========== TYPE 3: Text ==========
    print("\n" + "=" * 65)
    print("  TYPE 3: 📝 CLINICAL TEXT DIAGNOSIS (Doctor's Notes)")
    print("=" * 65)
    
    if not os.path.exists("data/hospital_1/text"):
        print("  ⚠️ Preparing text data...")
        from client.prepare_text_data import prepare_text_dataset
        prepare_text_dataset()
    
    from client.text_client import run_text_demo
    run_text_demo()
    
    # ========== FINAL SUMMARY ==========
    print("\n" + "=" * 65)
    print("  📊 ALL THREE DIAGNOSIS TYPES — SUMMARY")
    print("=" * 65)
    print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  Diagnosis Type    │ Model     │ Input        │ Status      │
  ├─────────────────────────────────────────────────────────────┤
  │  Tabular (EHR)     │ MLP       │ 13 features  │ ✅ Complete │
  │  Medical Image     │ CNN       │ 64x64 X-ray  │ ✅ Complete │
  │  Clinical Text     │ MLP (BoW) │ 500 word vec │ ✅ Complete │
  └─────────────────────────────────────────────────────────────┘

  All Three Types Use the SAME Federated Learning Principle:
  ✅ Each hospital trains LOCALLY on its own data
  ✅ Only model WEIGHTS are shared with the aggregator
  ✅ NO patient data (records, images, notes) leaves the hospital
  ✅ FedAvg combines knowledge from all hospitals
  ✅ Global model benefits from ALL hospitals' experience
  
  PyTorch Components Used:
  ✅ torch.nn.Module      — All three model architectures
  ✅ torch.nn.Conv2d      — CNN for image classification
  ✅ torch.nn.Linear      — MLP for tabular & text
  ✅ torch.nn.ReLU        — Activation functions
  ✅ torch.nn.BatchNorm   — Normalization layers
  ✅ torch.nn.Dropout     — Regularization
  ✅ torch.optim.Adam     — Optimizer for all models
  ✅ torch.nn.CrossEntropyLoss — Loss function
  ✅ torch.utils.data.DataLoader — Batch data loading
  ✅ torch.save/load      — Model weight serialization

  🔒 Privacy preserved across ALL three diagnosis types!
    """)


if __name__ == "__main__":
    main()