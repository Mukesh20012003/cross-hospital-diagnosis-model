# server/ehr_models.py

"""
PostgreSQL Database Models for EHR (Electronic Health Records)
================================================================
These tables store the REAL Heart Disease UCI dataset.
Each hospital has its own partition of patient records.

Tables:
  - patient_records    → Individual patient clinical data
  - hospital_datasets  → Dataset metadata per hospital
  - feature_statistics → Statistical summary per hospital
  
This is SEPARATE from the SQLite database which stores
federated learning metadata (rounds, models, audit logs).
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from server.postgres_config import PostgresBase


class HospitalDataset(PostgresBase):
    """Metadata about each hospital's local dataset"""
    __tablename__ = "hospital_datasets"

    id = Column(Integer, primary_key=True, index=True)
    hospital_name = Column(String(100), unique=True, nullable=False)
    dataset_source = Column(String(255), default="Cleveland Heart Disease (UCI)")
    dataset_type = Column(String(50), default="Tabular EHR")
    total_records = Column(Integer, default=0)
    disease_count = Column(Integer, default=0)
    healthy_count = Column(Integer, default=0)
    disease_ratio = Column(Float, default=0.0)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    records = relationship("PatientRecord", back_populates="hospital_dataset")
    statistics = relationship("FeatureStatistic", back_populates="hospital_dataset")

    def __repr__(self):
        return f"<HospitalDataset(name='{self.hospital_name}', records={self.total_records})>"


class PatientRecord(PostgresBase):
    """
    Individual patient clinical record.
    
    These are REAL clinical features from the UCI Heart Disease dataset.
    In a production hospital, this would connect to the hospital's EHR system
    (e.g., Epic, Cerner) via HL7 FHIR API.
    
    13 Features:
      age, sex, cp, trestbps, chol, fbs, restecg,
      thalach, exang, oldpeak, slope, ca, thal
    
    Target: 
      0 = No heart disease
      1 = Heart disease present
    """
    __tablename__ = "patient_records"

    id = Column(Integer, primary_key=True, index=True)
    hospital_dataset_id = Column(Integer, ForeignKey("hospital_datasets.id"), nullable=False)
    
    # ---- 13 Clinical Features ----
    age = Column(Float, nullable=False, comment="Age in years")
    sex = Column(Float, nullable=False, comment="Sex: 1=male, 0=female")
    cp = Column(Float, nullable=False, comment="Chest pain type: 0-3")
    trestbps = Column(Float, nullable=False, comment="Resting blood pressure (mm Hg)")
    chol = Column(Float, nullable=False, comment="Serum cholesterol (mg/dl)")
    fbs = Column(Float, nullable=False, comment="Fasting blood sugar > 120 mg/dl")
    restecg = Column(Float, nullable=False, comment="Resting ECG results: 0-2")
    thalach = Column(Float, nullable=False, comment="Maximum heart rate achieved")
    exang = Column(Float, nullable=False, comment="Exercise induced angina: 0/1")
    oldpeak = Column(Float, nullable=False, comment="ST depression induced by exercise")
    slope = Column(Float, nullable=False, comment="Slope of peak exercise ST segment")
    ca = Column(Float, nullable=False, comment="Number of major vessels: 0-3")
    thal = Column(Float, nullable=False, comment="Thalassemia: 1=normal, 2=fixed, 3=reversible")
    
    # ---- Diagnosis Target ----
    target = Column(Integer, nullable=False, comment="0=healthy, 1=heart disease")
    
    # ---- Metadata ----
    record_created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    hospital_dataset = relationship("HospitalDataset", back_populates="records")

    def __repr__(self):
        return f"<PatientRecord(id={self.id}, age={self.age}, target={self.target})>"
    
    def to_feature_list(self):
        """Convert record to list of feature values"""
        return [
            self.age, self.sex, self.cp, self.trestbps, self.chol,
            self.fbs, self.restecg, self.thalach, self.exang,
            self.oldpeak, self.slope, self.ca, self.thal
        ]


class FeatureStatistic(PostgresBase):
    """Statistical summary of features per hospital"""
    __tablename__ = "feature_statistics"

    id = Column(Integer, primary_key=True, index=True)
    hospital_dataset_id = Column(Integer, ForeignKey("hospital_datasets.id"), nullable=False)
    feature_name = Column(String(50), nullable=False)
    mean_value = Column(Float)
    std_value = Column(Float)
    min_value = Column(Float)
    max_value = Column(Float)
    
    # Relationships
    hospital_dataset = relationship("HospitalDataset", back_populates="statistics")

    def __repr__(self):
        return f"<FeatureStatistic(feature='{self.feature_name}', mean={self.mean_value})>"