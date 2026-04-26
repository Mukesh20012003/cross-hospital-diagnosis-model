# client/prepare_data.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import json


def download_heart_disease_data():
    """
    Download Heart Disease dataset.
    13 features, binary classification (disease / no disease)
    """
    print("📥 Downloading Heart Disease dataset...")
    
    # Use sklearn's built-in dataset
    from sklearn.datasets import load_wine
    
    # We'll create a synthetic heart disease-like dataset
    # that works reliably without internet issues
    np.random.seed(42)
    n_samples = 900
    
    # Generate 13 features (similar to heart disease features)
    feature_names = [
        "age", "sex", "chest_pain", "blood_pressure", "cholesterol",
        "blood_sugar", "ecg_result", "max_heart_rate", "exercise_angina",
        "st_depression", "st_slope", "num_vessels", "thalassemia"
    ]
    
    # Generate realistic medical data
    data = {
        "age": np.random.randint(29, 77, n_samples),
        "sex": np.random.randint(0, 2, n_samples),
        "chest_pain": np.random.randint(0, 4, n_samples),
        "blood_pressure": np.random.randint(94, 200, n_samples),
        "cholesterol": np.random.randint(126, 564, n_samples),
        "blood_sugar": np.random.randint(0, 2, n_samples),
        "ecg_result": np.random.randint(0, 3, n_samples),
        "max_heart_rate": np.random.randint(71, 202, n_samples),
        "exercise_angina": np.random.randint(0, 2, n_samples),
        "st_depression": np.random.uniform(0, 6.2, n_samples),
        "st_slope": np.random.randint(0, 3, n_samples),
        "num_vessels": np.random.randint(0, 4, n_samples),
        "thalassemia": np.random.randint(0, 3, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create target based on feature combinations (realistic correlation)
    risk_score = (
        (df["age"] > 55).astype(int) * 0.3 +
        (df["chest_pain"] >= 2).astype(int) * 0.2 +
        (df["blood_pressure"] > 140).astype(int) * 0.15 +
        (df["cholesterol"] > 300).astype(int) * 0.15 +
        (df["max_heart_rate"] < 120).astype(int) * 0.1 +
        (df["exercise_angina"] == 1).astype(int) * 0.1 +
        np.random.uniform(0, 0.3, n_samples)  # Add some noise
    )
    
    df["target"] = (risk_score > 0.5).astype(int)
    
    print(f"✅ Dataset created: {len(df)} samples, {len(feature_names)} features")
    print(f"   Class distribution: {dict(df['target'].value_counts())}")
    
    return df, feature_names


def split_into_hospitals(df, num_hospitals=3):
    """
    Split dataset into partitions for each hospital.
    Simulates each hospital having its own local data.
    """
    print(f"\n🏥 Splitting data into {num_hospitals} hospital partitions...")
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split size
    split_size = len(df) // num_hospitals
    
    hospital_data = {}
    for i in range(num_hospitals):
        hospital_name = f"hospital_{i+1}"
        
        # Split the dataframe
        start_idx = i * split_size
        if i == num_hospitals - 1:
            # Last hospital gets remaining data
            partition = df.iloc[start_idx:].copy()
        else:
            partition = df.iloc[start_idx:start_idx + split_size].copy()
        
        partition = partition.reset_index(drop=True)
        hospital_data[hospital_name] = partition
        
        print(f"   {hospital_name}: {len(partition)} samples, "
              f"Disease: {partition['target'].sum()}, "
              f"Healthy: {len(partition) - partition['target'].sum()}")
    
    return hospital_data


def preprocess_and_save(hospital_data, feature_names):
    """
    Preprocess data and save as PyTorch tensors for each hospital.
    """
    print("\n💾 Preprocessing and saving data...")
    
    # Fit scaler on ALL data combined (in real scenario, each hospital would scale independently)
    all_data = pd.concat(hospital_data.values())
    scaler = StandardScaler()
    scaler.fit(all_data[feature_names])
    
    # Also create a global test set
    test_size = 50  # Small test set for server-side evaluation
    
    for hospital_name, data in hospital_data.items():
        # Create directory
        hospital_dir = os.path.join("data", hospital_name)
        os.makedirs(hospital_dir, exist_ok=True)
        
        # Separate features and target
        X = data[feature_names].values
        y = data["target"].values
        
        # Scale features
        X_scaled = scaler.transform(pd.DataFrame(X, columns=feature_names))
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Convert to PyTorch tensors
        train_data = {
            "X_train": torch.FloatTensor(X_train),
            "y_train": torch.LongTensor(y_train),
            "X_val": torch.FloatTensor(X_val),
            "y_val": torch.LongTensor(y_val)
        }
        
        # Save tensors
        torch.save(train_data, os.path.join(hospital_dir, "data.pt"))
        
        # Save metadata
        metadata = {
            "hospital_name": hospital_name,
            "total_samples": len(data),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "num_features": len(feature_names),
            "feature_names": feature_names,
            "class_distribution": {
                "disease": int(data["target"].sum()),
                "healthy": int(len(data) - data["target"].sum())
            }
        }
        
        with open(os.path.join(hospital_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ {hospital_name}: Saved {len(X_train)} train, {len(X_val)} val samples")
    
    # Create global test set for server evaluation
    test_dir = os.path.join("data", "test_set")
    os.makedirs(test_dir, exist_ok=True)
    
    all_X = all_data[feature_names].values
    all_y = all_data["target"].values
    all_X_scaled = scaler.transform(pd.DataFrame(all_X, columns=feature_names))
    
    # Take last few samples as global test
    test_data = {
        "X_test": torch.FloatTensor(all_X_scaled[-test_size:]),
        "y_test": torch.LongTensor(all_y[-test_size:].copy())
    }
    torch.save(test_data, os.path.join(test_dir, "test_data.pt"))
    print(f"   ✅ Global test set: {test_size} samples saved")
    
    print("\n🎉 Data preparation complete!")


def main():
    """Main function to prepare all data"""
    print("=" * 60)
    print("  CROSS-HOSPITAL DIAGNOSIS - DATA PREPARATION")
    print("=" * 60)
    
    # Step 1: Download/create dataset
    df, feature_names = download_heart_disease_data()
    
    # Step 2: Split into hospital partitions
    hospital_data = split_into_hospitals(df, num_hospitals=3)
    
    # Step 3: Preprocess and save
    preprocess_and_save(hospital_data, feature_names)
    
    print("\n" + "=" * 60)
    print("  DATA FILES CREATED:")
    print("=" * 60)
    print("  data/hospital_1/data.pt       - Hospital 1 training data")
    print("  data/hospital_1/metadata.json  - Hospital 1 metadata")
    print("  data/hospital_2/data.pt       - Hospital 2 training data")
    print("  data/hospital_2/metadata.json  - Hospital 2 metadata")
    print("  data/hospital_3/data.pt       - Hospital 3 training data")
    print("  data/hospital_3/metadata.json  - Hospital 3 metadata")
    print("  data/test_set/test_data.pt    - Global test set")
    print("=" * 60)


if __name__ == "__main__":
    main()