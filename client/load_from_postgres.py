# client/load_from_postgres.py

"""
Load Hospital Training Data from PostgreSQL
=============================================
Alternative data loader that reads from PostgreSQL instead of .pt files.
Hospital clients can use EITHER source:
  - Existing: torch.load("data/hospital_X/data.pt")
  - New:      load_from_postgres("hospital_1")

Both produce the same format: {X_train, y_train, X_val, y_val}
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from server.postgres_config import PostgresSession
from server.ehr_models import HospitalDataset, PatientRecord, FeatureStatistic


FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def load_from_postgres(hospital_name: str) -> dict:
    """
    Load training data for a hospital from PostgreSQL.
    
    This simulates a hospital loading data from its local EHR database.
    In production, this would connect to the hospital's own database.
    
    Args:
        hospital_name: e.g., "hospital_1"
    
    Returns:
        dict with X_train, y_train, X_val, y_val (PyTorch tensors)
    """
    print(f"\n  📂 Loading data from PostgreSQL for {hospital_name}...")
    
    session = PostgresSession()
    
    try:
        # Find hospital dataset
        hospital_ds = session.query(HospitalDataset).filter(
            HospitalDataset.hospital_name == hospital_name
        ).first()
        
        if not hospital_ds:
            print(f"  ❌ No data found for {hospital_name} in PostgreSQL!")
            print(f"  Run 'python -m client.prepare_real_data' first.")
            return None
        
        # Load patient records
        records = session.query(PatientRecord).filter(
            PatientRecord.hospital_dataset_id == hospital_ds.id
        ).all()
        
        if not records:
            print(f"  ❌ No patient records found for {hospital_name}!")
            return None
        
        # Convert to numpy arrays
        X = np.array([r.to_feature_list() for r in records], dtype=np.float32)
        y = np.array([r.target for r in records], dtype=np.int64)
        
        print(f"  ✅ Loaded {len(records)} records from PostgreSQL")
        print(f"     Disease: {y.sum()} | Healthy: {len(y) - y.sum()}")
        
        # Preprocess: Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Convert to PyTorch tensors
        data = {
            "X_train": torch.FloatTensor(X_train),
            "y_train": torch.LongTensor(y_train),
            "X_val": torch.FloatTensor(X_val),
            "y_val": torch.LongTensor(y_val)
        }
        
        print(f"  ✅ Preprocessed: {len(X_train)} train, {len(X_val)} val samples")
        
        return data
    
    finally:
        session.close()


def load_test_set_from_postgres() -> dict:
    """Load global test set from PostgreSQL"""
    
    session = PostgresSession()
    
    try:
        hospital_ds = session.query(HospitalDataset).filter(
            HospitalDataset.hospital_name == "test_set"
        ).first()
        
        if not hospital_ds:
            print("  ❌ No test set found in PostgreSQL!")
            return None
        
        records = session.query(PatientRecord).filter(
            PatientRecord.hospital_dataset_id == hospital_ds.id
        ).all()
        
        X = np.array([r.to_feature_list() for r in records], dtype=np.float32)
        y = np.array([r.target for r in records], dtype=np.int64)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return {
            "X_test": torch.FloatTensor(X_scaled),
            "y_test": torch.LongTensor(y)
        }
    
    finally:
        session.close()


def get_hospital_metadata(hospital_name: str) -> dict:
    """Get dataset metadata from PostgreSQL"""
    
    session = PostgresSession()
    
    try:
        hospital_ds = session.query(HospitalDataset).filter(
            HospitalDataset.hospital_name == hospital_name
        ).first()
        
        if not hospital_ds:
            return None
        
        stats = session.query(FeatureStatistic).filter(
            FeatureStatistic.hospital_dataset_id == hospital_ds.id
        ).all()
        
        return {
            "hospital_name": hospital_ds.hospital_name,
            "source": hospital_ds.dataset_source,
            "type": hospital_ds.dataset_type,
            "total_records": hospital_ds.total_records,
            "disease_count": hospital_ds.disease_count,
            "healthy_count": hospital_ds.healthy_count,
            "disease_ratio": hospital_ds.disease_ratio,
            "features": {
                s.feature_name: {
                    "mean": s.mean_value,
                    "std": s.std_value,
                    "min": s.min_value,
                    "max": s.max_value
                } for s in stats
            }
        }
    
    finally:
        session.close()


# ============================================
# ALSO SAVE AS .PT FILES (for compatibility)
# ============================================
def postgres_to_pt_files():
    """
    Export PostgreSQL data to .pt files so existing code works.
    This bridges the new PostgreSQL data with the existing pipeline.
    """
    print("\n" + "=" * 50)
    print("  📦 EXPORTING PostgreSQL → .pt FILES")
    print("=" * 50)
    
    for i in range(1, 4):
        hospital_name = f"hospital_{i}"
        data = load_from_postgres(hospital_name)
        
        if data:
            hospital_dir = os.path.join("data", hospital_name)
            os.makedirs(hospital_dir, exist_ok=True)
            
            # Save as .pt file (compatible with existing code)
            torch.save(data, os.path.join(hospital_dir, "data.pt"))
            
            # Save metadata
            import json
            metadata = get_hospital_metadata(hospital_name)
            if metadata:
                metadata["data_source"] = "PostgreSQL (Real UCI Heart Disease Dataset)"
                metadata["privacy_note"] = "Data remains on-premises. Only model weights are shared."
                with open(os.path.join(hospital_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            print(f"  ✅ {hospital_name}: data.pt + metadata.json saved")
    
    # Test set
    test_data = load_test_set_from_postgres()
    if test_data:
        test_dir = os.path.join("data", "test_set")
        os.makedirs(test_dir, exist_ok=True)
        torch.save(test_data, os.path.join(test_dir, "test_data.pt"))
        print(f"  ✅ test_set: test_data.pt saved")
    
    print("\n  ✅ All .pt files created from PostgreSQL data!")
    print("  Existing federated learning pipeline will use this data.")


if __name__ == "__main__":
    # Demo: Load from PostgreSQL and export to .pt
    print("=" * 60)
    print("  POSTGRESQL DATA LOADER DEMO")
    print("=" * 60)
    
    postgres_to_pt_files()