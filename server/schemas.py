# server/schemas.py

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


# ---- Hospital Schemas ----
class HospitalRegister(BaseModel):
    name: str
    location: Optional[str] = None
    data_size: Optional[int] = 0


class HospitalResponse(BaseModel):
    id: int
    name: str
    api_key: str
    location: Optional[str]
    data_size: int
    is_active: bool
    registered_at: datetime
    last_seen: datetime

    class Config:
        from_attributes = True


# ---- Training Round Schemas ----
class RoundCreate(BaseModel):
    target_participants: Optional[int] = 3


class RoundResponse(BaseModel):
    id: int
    round_number: int
    status: str
    num_participants: int
    target_participants: int
    global_accuracy: Optional[float]
    global_loss: Optional[float]
    started_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


# ---- Model Update Schemas ----
class UpdateSubmitResponse(BaseModel):
    message: str
    round_number: int
    hospital_name: str
    updates_received: int
    target: int


# ---- Global Model Schemas ----
class GlobalModelResponse(BaseModel):
    id: int
    round_id: int
    version: str
    accuracy: Optional[float]
    loss: Optional[float]
    created_at: datetime
    download_count: int

    class Config:
        from_attributes = True


# ---- Audit Log Schemas ----
class AuditLogResponse(BaseModel):
    id: int
    hospital_id: Optional[int]
    action: str
    details: Optional[str]
    ip_address: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True


# ---- Dashboard Schemas ----
class DashboardStats(BaseModel):
    total_hospitals: int
    active_hospitals: int
    current_round: int
    total_rounds: int
    latest_accuracy: Optional[float]
    total_updates: int
    model_downloads: int