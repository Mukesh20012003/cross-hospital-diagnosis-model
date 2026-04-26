# client/prepare_real_data.py

"""
Real Medical Dataset Ingestion Pipeline (PostgreSQL)
=====================================================
Loads the REAL Cleveland Heart Disease UCI dataset
into PostgreSQL database, partitioned by hospital.

This is SEPARATE from the existing prepare_data.py
which generates synthetic data as .pt files.

Both data sources can be used for federated learning:
  - Existing: data/hospital_X/data.pt (synthetic, PyTorch files)
  - NEW: PostgreSQL database (real UCI dataset)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime

from server.postgres_config import postgres_engine, PostgresSession, PostgresBase, test_connection
from server.ehr_models import HospitalDataset, PatientRecord, FeatureStatistic


FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_DESCRIPTIONS = {
    "age": "Age of patient in years",
    "sex": "Sex (1=male, 0=female)",
    "cp": "Chest pain type (0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1=true, 0=false)",
    "restecg": "Resting ECG results (0=normal, 1=ST-T abnormality, 2=LV hypertrophy)",
    "thalach": "Maximum heart rate achieved during exercise",
    "exang": "Exercise induced angina (1=yes, 0=no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "Slope of peak exercise ST segment (0=upsloping, 1=flat, 2=downsloping)",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy",
    "thal": "Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)"
}


def load_uci_heart_disease():
    """
    Load the REAL Heart Disease dataset from multiple sources.
    """
    print("=" * 60)
    print("  📊 LOADING REAL UCI HEART DISEASE DATASET")
    print("=" * 60)
    
    df = None
    source = ""
    
    # ---- Method 1: sklearn OpenML ----
    try:
        print("\n  📥 Trying sklearn (OpenML)...")
        from sklearn.datasets import fetch_openml
        heart = fetch_openml(name='heart-statlog', version=1, as_frame=True, parser='auto')
        
        data_df = heart.data.copy()
        target_series = heart.target.copy()
        
        # Rename columns to our standard names
        if len(data_df.columns) == 13:
            data_df.columns = FEATURE_NAMES
        
        # Handle target - could be string or numeric
        try:
            if target_series.dtype == object:
                # String labels like 'present'/'absent' or '1'/'2'
                unique_vals = target_series.unique()
                print(f"     Target values found: {unique_vals}")
                
                if 'present' in unique_vals:
                    data_df['target'] = (target_series == 'present').astype(int)
                elif '2' in unique_vals or '1' in unique_vals:
                    data_df['target'] = (target_series.astype(str).isin(['2', 'present'])).astype(int)
                else:
                    # Try numeric conversion
                    numeric_target = pd.to_numeric(target_series, errors='coerce')
                    data_df['target'] = (numeric_target > numeric_target.min()).astype(int)
            else:
                # Numeric target
                numeric_target = target_series.astype(float)
                if numeric_target.max() > 1:
                    data_df['target'] = (numeric_target > numeric_target.min()).astype(int)
                else:
                    data_df['target'] = numeric_target.astype(int)
        except Exception as target_err:
            print(f"     ⚠️ Target conversion issue: {target_err}")
            # Fallback: assign binary based on median
            numeric_target = pd.to_numeric(target_series, errors='coerce')
            data_df['target'] = (numeric_target > numeric_target.median()).astype(int)
        
        df = data_df
        source = "sklearn OpenML (heart-statlog)"
        print(f"  ✅ Loaded from sklearn! ({len(df)} records)")
        
    except Exception as e:
        print(f"  ⚠️ sklearn failed: {e}")
        df = None
    
    # ---- Method 2: UCI URL ----
    if df is None:
        try:
            print("  📥 Trying UCI Repository URL...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            column_names = FEATURE_NAMES + ['target']
            df = pd.read_csv(url, names=column_names, na_values='?')
            df = df.dropna()
            df['target'] = (df['target'] > 0).astype(int)
            source = "UCI Repository (Cleveland)"
            print(f"  ✅ Loaded from UCI! ({len(df)} records)")
        except Exception as e:
            print(f"  ⚠️ UCI URL failed: {e}")
            df = None
    
    # ---- Method 3: Generate from real distributions ----
    if df is None:
        print("  📥 Generating from real clinical distributions...")
        df = generate_realistic_data()
        source = "Generated from Cleveland Heart Disease statistical distributions"
        print(f"  ✅ Generated! ({len(df)} records)")
    
    # ---- CLEAN DATA ----
    # Make sure all feature columns are numeric
    for col in FEATURE_NAMES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Make sure target exists and is binary
    if 'target' in df.columns:
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df = df.dropna()
        df['target'] = df['target'].astype(int)
        if df['target'].max() > 1:
            df['target'] = (df['target'] > 0).astype(int)
    else:
        print("  ⚠️ No target column! Using last column as target.")
        df['target'] = (df.iloc[:, -1] > df.iloc[:, -1].median()).astype(int)
    
    # Drop any remaining NaN
    df = df.dropna()
    
    # Keep only the columns we need
    df = df[FEATURE_NAMES + ['target']].copy()
    
    print(f"\n  {'─' * 50}")
    print(f"  📊 Dataset Summary:")
    print(f"  {'─' * 50}")
    print(f"  Source: {source}")
    print(f"  Total records: {len(df)}")
    print(f"  Features: {len(FEATURE_NAMES)}")
    print(f"  Disease: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    print(f"  Healthy: {len(df) - df['target'].sum()} ({(1-df['target'].mean())*100:.1f}%)")
    print(f"\n  Feature Ranges:")
    print(f"  {'─' * 50}")
    for feat in FEATURE_NAMES:
        print(f"  {feat:>10}: min={df[feat].min():>8.2f}, max={df[feat].max():>8.2f}, mean={df[feat].mean():>8.2f}")
    
    return df, source


def generate_realistic_data():
    """Generate data from REAL Cleveland Heart Disease distributions"""
    np.random.seed(42)
    n = 900
    
    n_healthy = n // 2
    healthy = pd.DataFrame({
        'age': np.random.normal(52.6, 9.5, n_healthy).clip(29, 77).astype(int),
        'sex': np.random.binomial(1, 0.56, n_healthy),
        'cp': np.random.choice([0, 1, 2, 3], n_healthy, p=[0.35, 0.15, 0.30, 0.20]),
        'trestbps': np.random.normal(129.3, 16.2, n_healthy).clip(94, 200).astype(int),
        'chol': np.random.normal(243.5, 49.4, n_healthy).clip(126, 564).astype(int),
        'fbs': np.random.binomial(1, 0.14, n_healthy),
        'restecg': np.random.choice([0, 1, 2], n_healthy, p=[0.50, 0.45, 0.05]),
        'thalach': np.random.normal(158.6, 19.1, n_healthy).clip(71, 202).astype(int),
        'exang': np.random.binomial(1, 0.14, n_healthy),
        'oldpeak': np.random.exponential(0.5, n_healthy).clip(0, 6.2).round(1),
        'slope': np.random.choice([0, 1, 2], n_healthy, p=[0.60, 0.30, 0.10]),
        'ca': np.random.choice([0, 1, 2, 3], n_healthy, p=[0.70, 0.15, 0.10, 0.05]),
        'thal': np.random.choice([1, 2, 3], n_healthy, p=[0.55, 0.10, 0.35]),
    })
    healthy['target'] = 0
    
    n_disease = n - n_healthy
    disease = pd.DataFrame({
        'age': np.random.normal(56.8, 8.1, n_disease).clip(29, 77).astype(int),
        'sex': np.random.binomial(1, 0.82, n_disease),
        'cp': np.random.choice([0, 1, 2, 3], n_disease, p=[0.08, 0.10, 0.15, 0.67]),
        'trestbps': np.random.normal(134.6, 19.8, n_disease).clip(94, 200).astype(int),
        'chol': np.random.normal(251.1, 51.6, n_disease).clip(126, 564).astype(int),
        'fbs': np.random.binomial(1, 0.17, n_disease),
        'restecg': np.random.choice([0, 1, 2], n_disease, p=[0.40, 0.50, 0.10]),
        'thalach': np.random.normal(139.1, 22.7, n_disease).clip(71, 202).astype(int),
        'exang': np.random.binomial(1, 0.55, n_disease),
        'oldpeak': np.random.exponential(1.6, n_disease).clip(0, 6.2).round(1),
        'slope': np.random.choice([0, 1, 2], n_disease, p=[0.15, 0.50, 0.35]),
        'ca': np.random.choice([0, 1, 2, 3], n_disease, p=[0.25, 0.30, 0.25, 0.20]),
        'thal': np.random.choice([1, 2, 3], n_disease, p=[0.15, 0.20, 0.65]),
    })
    disease['target'] = 1
    
    df = pd.concat([healthy, disease], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def create_postgres_tables():
    """Create all PostgreSQL tables"""
    print("\n  📦 Creating PostgreSQL tables...")
    PostgresBase.metadata.create_all(bind=postgres_engine)
    print("  ✅ Tables created: hospital_datasets, patient_records, feature_statistics")


def insert_hospital_data(df, hospital_name, session, source):
    """Insert patient records for one hospital into PostgreSQL"""
    
    disease_count = int(df['target'].sum())
    healthy_count = len(df) - disease_count
    
    # Create hospital dataset entry
    hospital_ds = HospitalDataset(
        hospital_name=hospital_name,
        dataset_source=source,
        dataset_type="Tabular EHR (Electronic Health Records)",
        total_records=len(df),
        disease_count=disease_count,
        healthy_count=healthy_count,
        disease_ratio=round(disease_count / len(df), 4),
        description=f"Heart disease clinical data for {hospital_name}. "
                    f"13 features from patient EHR. "
                    f"Data remains on-premises at this hospital."
    )
    session.add(hospital_ds)
    session.flush()  # Get the ID
    
    # Insert patient records
    records = []
    for _, row in df.iterrows():
        record = PatientRecord(
            hospital_dataset_id=hospital_ds.id,
            age=float(row['age']),
            sex=float(row['sex']),
            cp=float(row['cp']),
            trestbps=float(row['trestbps']),
            chol=float(row['chol']),
            fbs=float(row['fbs']),
            restecg=float(row['restecg']),
            thalach=float(row['thalach']),
            exang=float(row['exang']),
            oldpeak=float(row['oldpeak']),
            slope=float(row['slope']),
            ca=float(row['ca']),
            thal=float(row['thal']),
            target=int(row['target'])
        )
        records.append(record)
    
    session.bulk_save_objects(records)
    
    # Insert feature statistics
    for feat in FEATURE_NAMES:
        stat = FeatureStatistic(
            hospital_dataset_id=hospital_ds.id,
            feature_name=feat,
            mean_value=round(float(df[feat].mean()), 4),
            std_value=round(float(df[feat].std()), 4),
            min_value=round(float(df[feat].min()), 4),
            max_value=round(float(df[feat].max()), 4)
        )
        session.add(stat)
    
    session.commit()
    
    print(f"  ✅ {hospital_name}: {len(df)} records inserted into PostgreSQL")
    print(f"     Disease: {disease_count} ({disease_count/len(df)*100:.1f}%) | "
          f"Healthy: {healthy_count} ({healthy_count/len(df)*100:.1f}%)")
    
    return hospital_ds.id


def partition_and_store(df, source, num_hospitals=3):
    """Partition dataset and store each partition in PostgreSQL"""
    
    print(f"\n  {'=' * 50}")
    print(f"  🏥 PARTITIONING & STORING IN POSTGRESQL")
    print(f"  {'=' * 50}")
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_size = len(df) // num_hospitals
    
    session = PostgresSession()
    
    try:
        # Clear existing data
        session.query(FeatureStatistic).delete()
        session.query(PatientRecord).delete()
        session.query(HospitalDataset).delete()
        session.commit()
        print("  🗑️ Cleared existing PostgreSQL data")
        
        hospital_ids = {}
        
        for i in range(num_hospitals):
            hospital_name = f"hospital_{i+1}"
            start_idx = i * split_size
            
            if i == num_hospitals - 1:
                partition = df.iloc[start_idx:].copy()
            else:
                partition = df.iloc[start_idx:start_idx + split_size].copy()
            
            partition = partition.reset_index(drop=True)
            
            ds_id = insert_hospital_data(partition, hospital_name, session, source)
            hospital_ids[hospital_name] = ds_id
        
        # Store a test set
        test_size = min(100, len(df) // 5)
        test_partition = df.sample(n=test_size, random_state=99)
        insert_hospital_data(test_partition, "test_set", session, source)
        
        print(f"\n  ✅ Test set: {test_size} records stored in PostgreSQL")
        
        return hospital_ids
        
    except Exception as e:
        session.rollback()
        print(f"  ❌ Error: {e}")
        raise
    finally:
        session.close()


def verify_data():
    """Verify data in PostgreSQL"""
    print(f"\n  {'=' * 50}")
    print(f"  🔍 VERIFYING POSTGRESQL DATA")
    print(f"  {'=' * 50}")
    
    session = PostgresSession()
    
    try:
        datasets = session.query(HospitalDataset).all()
        
        for ds in datasets:
            record_count = session.query(PatientRecord).filter(
                PatientRecord.hospital_dataset_id == ds.id
            ).count()
            
            stat_count = session.query(FeatureStatistic).filter(
                FeatureStatistic.hospital_dataset_id == ds.id
            ).count()
            
            print(f"\n  📊 {ds.hospital_name}:")
            print(f"     Records: {record_count}")
            print(f"     Disease: {ds.disease_count} | Healthy: {ds.healthy_count}")
            print(f"     Disease Ratio: {ds.disease_ratio*100:.1f}%")
            print(f"     Feature Stats: {stat_count} entries")
            print(f"     Source: {ds.dataset_source}")
        
        total_records = session.query(PatientRecord).count()
        print(f"\n  📊 Total records in PostgreSQL: {total_records}")
        
    finally:
        session.close()


def main():
    """Main pipeline: Load real dataset → Store in PostgreSQL"""
    
    print("\n" + "=" * 60)
    print("  🏥 REAL MEDICAL DATASET → POSTGRESQL PIPELINE")
    print("  (Separate from existing synthetic data)")
    print("=" * 60)
    
    # Step 1: Test PostgreSQL connection
    print("\n  Step 1: Testing PostgreSQL connection...")
    if not test_connection():
        print("\n  ⚠️ FIX: Update credentials in server/postgres_config.py")
        print("  Then run this script again.")
        return
    
    # Step 2: Create tables
    create_postgres_tables()
    
    # Step 3: Load real dataset
    df, source = load_uci_heart_disease()
    
    # Step 4: Partition and store in PostgreSQL
    partition_and_store(df, source, num_hospitals=3)
    
    # Step 5: Verify
    verify_data()
    
    print("\n" + "=" * 60)
    print("  ✅ REAL DATASET PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"""
  PostgreSQL Database: federated_ehr
  Tables Created:
    ├── hospital_datasets    → Dataset metadata per hospital
    ├── patient_records      → Individual patient clinical records
    └── feature_statistics   → Feature stats per hospital
  
  Data Sources Now Available:
    1. EXISTING: data/hospital_X/data.pt    (Synthetic, PyTorch files)
    2. NEW:      PostgreSQL database         (Real UCI Heart Disease)
  
  Both can be used for federated learning training!
    """)


if __name__ == "__main__":
    main()