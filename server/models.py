# server/models.py

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from server.database import Base


class Hospital(Base):
    """Registered hospital nodes"""
    __tablename__ = "hospitals"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    api_key = Column(String(255), unique=True, nullable=False)
    location = Column(String(200), nullable=True)
    data_size = Column(Integer, default=0)  # Number of samples at this hospital
    is_active = Column(Boolean, default=True)
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)

    # Relationships
    updates = relationship("ModelUpdate", back_populates="hospital")
    audit_logs = relationship("AuditLog", back_populates="hospital")

    def __repr__(self):
        return f"<Hospital(name='{self.name}', active={self.is_active})>"


class TrainingRound(Base):
    """Each federated learning round"""
    __tablename__ = "training_rounds"

    id = Column(Integer, primary_key=True, index=True)
    round_number = Column(Integer, unique=True, nullable=False)
    status = Column(String(20), default="in_progress")  # in_progress, completed, failed
    num_participants = Column(Integer, default=0)
    target_participants = Column(Integer, default=3)
    global_accuracy = Column(Float, nullable=True)
    global_loss = Column(Float, nullable=True)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    updates = relationship("ModelUpdate", back_populates="training_round")
    global_model = relationship("GlobalModel", back_populates="training_round", uselist=False)

    def __repr__(self):
        return f"<TrainingRound(round={self.round_number}, status='{self.status}')>"


class ModelUpdate(Base):
    """Weight updates submitted by each hospital per round"""
    __tablename__ = "model_updates"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=False)
    round_id = Column(Integer, ForeignKey("training_rounds.id"), nullable=False)
    file_path = Column(String(500), nullable=False)  # Path to saved weights file
    data_size = Column(Integer, nullable=False)  # Samples used for training
    local_accuracy = Column(Float, nullable=True)
    local_loss = Column(Float, nullable=True)
    submitted_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    hospital = relationship("Hospital", back_populates="updates")
    training_round = relationship("TrainingRound", back_populates="updates")

    def __repr__(self):
        return f"<ModelUpdate(hospital={self.hospital_id}, round={self.round_id})>"


class GlobalModel(Base):
    """Aggregated global model after each round"""
    __tablename__ = "global_models"

    id = Column(Integer, primary_key=True, index=True)
    round_id = Column(Integer, ForeignKey("training_rounds.id"), nullable=False)
    file_path = Column(String(500), nullable=False)  # Path to saved global model
    accuracy = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)
    version = Column(String(50), nullable=False)  # e.g., "v1", "v2"
    created_at = Column(DateTime, default=datetime.utcnow)
    download_count = Column(Integer, default=0)

    # Relationships
    training_round = relationship("TrainingRound", back_populates="global_model")

    def __repr__(self):
        return f"<GlobalModel(version='{self.version}', accuracy={self.accuracy})>"


class AuditLog(Base):
    """Audit trail for compliance and security"""
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id"), nullable=True)
    action = Column(String(100), nullable=False)  # e.g., "submit_update", "download_model"
    details = Column(Text, nullable=True)
    ip_address = Column(String(50), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationships
    hospital = relationship("Hospital", back_populates="audit_logs")

    def __repr__(self):
        return f"<AuditLog(action='{self.action}', time={self.timestamp})>"