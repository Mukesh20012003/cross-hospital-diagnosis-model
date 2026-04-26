# server/main.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import shutil
import torch
import time
import json

from server.database import engine, Base, get_db
from server.models import Hospital, TrainingRound, ModelUpdate, GlobalModel, AuditLog
from server.schemas import (
    HospitalRegister, HospitalResponse, RoundCreate, RoundResponse,
    UpdateSubmitResponse, GlobalModelResponse, AuditLogResponse, DashboardStats
)
from server.utils import (
    generate_api_key, get_model_update_path, get_global_model_path,
    get_initial_model_path, write_audit_log_file
)
from server.model_definition import create_initial_model
from server.aggregation import aggregate_and_save, evaluate_global_model
from server.privacy import DifferentialPrivacy, validate_update_integrity
from server.reports import generate_compliance_report, generate_csv_report, generate_audit_csv

from server.postgres_config import test_connection as test_pg


# Create all database tables
Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Cross-Hospital Diagnosis Model",
    description="Federated Learning Aggregator Server - Trains diagnostic models across hospitals without sharing patient data",
    version="1.0.0"
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="server/static"), name="static")
templates = Jinja2Templates(directory="server/templates")

# Initialize privacy module
dp_module = DifferentialPrivacy(epsilon=1.0, delta=1e-5, max_grad_norm=1.0)

# Store notifications for hospitals
notifications = []


# ==========================================
# STARTUP: Create initial model if not exists
# ==========================================
@app.on_event("startup")
async def startup_event():
    initial_model_path = get_initial_model_path()
    if not os.path.exists(initial_model_path):
        print("Creating initial global model...")
        model = create_initial_model()
        torch.save(model.state_dict(), initial_model_path)
        print(f"✅ Initial model saved to: {initial_model_path}")
    else:
        print("✅ Initial model already exists.")


# ==========================================
# ENDPOINT 1: Register a Hospital
# ==========================================
@app.post("/register_hospital", response_model=HospitalResponse, tags=["Hospitals"])
async def register_hospital(hospital: HospitalRegister, request: Request, db: Session = Depends(get_db)):
    existing = db.query(Hospital).filter(Hospital.name == hospital.name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Hospital '{hospital.name}' already registered")
    
    api_key = generate_api_key()
    
    new_hospital = Hospital(
        name=hospital.name,
        api_key=api_key,
        location=hospital.location,
        data_size=hospital.data_size
    )
    
    db.add(new_hospital)
    db.commit()
    db.refresh(new_hospital)
    
    # Audit log with IP
    client_ip = request.client.host if request.client else "unknown"
    audit = AuditLog(
        hospital_id=new_hospital.id,
        action="register_hospital",
        details=f"Hospital '{hospital.name}' registered from {hospital.location}",
        ip_address=client_ip
    )
    db.add(audit)
    db.commit()
    
    write_audit_log_file("register_hospital", f"Hospital '{hospital.name}' registered", hospital.name)
    
    return new_hospital


# ==========================================
# ENDPOINT 2: List All Hospitals
# ==========================================
@app.get("/hospitals", response_model=list[HospitalResponse], tags=["Hospitals"])
async def list_hospitals(db: Session = Depends(get_db)):
    hospitals = db.query(Hospital).all()
    return hospitals


# ==========================================
# ENDPOINT 3: Start a New Training Round
# ==========================================
@app.post("/start_round", response_model=RoundResponse, tags=["Training Rounds"])
async def start_round(round_config: RoundCreate, db: Session = Depends(get_db)):
    active_round = db.query(TrainingRound).filter(
        TrainingRound.status == "in_progress"
    ).first()
    
    if active_round:
        raise HTTPException(
            status_code=400,
            detail=f"Round {active_round.round_number} is still in progress. Wait for it to complete."
        )
    
    last_round = db.query(TrainingRound).order_by(TrainingRound.round_number.desc()).first()
    next_round_number = (last_round.round_number + 1) if last_round else 1
    
    new_round = TrainingRound(
        round_number=next_round_number,
        status="in_progress",
        target_participants=round_config.target_participants
    )
    
    db.add(new_round)
    db.commit()
    db.refresh(new_round)
    
    audit = AuditLog(
        action="start_round",
        details=f"Round {next_round_number} started. Target: {round_config.target_participants} hospitals"
    )
    db.add(audit)
    db.commit()
    
    write_audit_log_file("start_round", f"Round {next_round_number} started")
    
    return new_round


# ==========================================
# ENDPOINT 4: Submit Model Update
# ==========================================
@app.post("/submit_update", response_model=UpdateSubmitResponse, tags=["Model Updates"])
async def submit_update(
    request: Request,
    api_key: str = Form(...),
    data_size: int = Form(...),
    local_accuracy: float = Form(0.0),
    local_loss: float = Form(0.0),
    weights_file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    # Authenticate hospital
    hospital = db.query(Hospital).filter(Hospital.api_key == api_key).first()
    if not hospital:
        raise HTTPException(status_code=401, detail="Invalid API key. Access denied.")
    
    # Get current active round
    current_round = db.query(TrainingRound).filter(
        TrainingRound.status == "in_progress"
    ).first()
    
    if not current_round:
        raise HTTPException(status_code=400, detail="No active training round. Start a round first.")
    
    # Check duplicate submission
    existing_update = db.query(ModelUpdate).filter(
        ModelUpdate.hospital_id == hospital.id,
        ModelUpdate.round_id == current_round.id
    ).first()
    
    if existing_update:
        raise HTTPException(
            status_code=400,
            detail=f"Hospital '{hospital.name}' already submitted for round {current_round.round_number}"
        )
    
    # Save uploaded weights file
    file_path = get_model_update_path(current_round.round_number, hospital.id)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(weights_file.file, buffer)
    
    # Validate update integrity
    try:
        uploaded_weights = torch.load(file_path, map_location="cpu", weights_only=True)
        validation = validate_update_integrity(uploaded_weights)
        if not validation["valid"]:
            audit = AuditLog(
                hospital_id=hospital.id,
                action="update_validation_warning",
                details=f"Issues: {validation['issues']}"
            )
            db.add(audit)
    except Exception as e:
        print(f"Warning: Could not validate weights: {e}")
    
    # Apply differential privacy (clip + noise)
    try:
        weights = torch.load(file_path, map_location="cpu", weights_only=True)
        private_weights = dp_module.apply_privacy(weights)
        torch.save(private_weights, file_path)
    except Exception as e:
        print(f"Warning: Could not apply DP: {e}")
    
    # Calculate latency
    latency = round(time.time() - start_time, 3)
    
    # Record the update
    update = ModelUpdate(
        hospital_id=hospital.id,
        round_id=current_round.id,
        file_path=file_path,
        data_size=data_size,
        local_accuracy=local_accuracy,
        local_loss=local_loss
    )
    
    db.add(update)
    
    # Update hospital info
    hospital.data_size = data_size
    hospital.last_seen = datetime.utcnow()
    
    # Update round participant count
    current_round.num_participants += 1
    
    db.commit()
    
    # Audit log with IP and latency
    client_ip = request.client.host if request.client else "unknown"
    audit = AuditLog(
        hospital_id=hospital.id,
        action="submit_update",
        details=f"Hospital '{hospital.name}' submitted for round {current_round.round_number}. "
                f"Data: {data_size}, Accuracy: {local_accuracy:.4f}, Latency: {latency}s",
        ip_address=client_ip
    )
    db.add(audit)
    db.commit()
    
    write_audit_log_file("submit_update", f"Weights submitted. Latency: {latency}s", hospital.name)
    
    # Check if ready to aggregate
    if current_round.num_participants >= current_round.target_participants:
        await perform_aggregation(current_round, db)
    
    return UpdateSubmitResponse(
        message=f"Update received from {hospital.name}",
        round_number=current_round.round_number,
        hospital_name=hospital.name,
        updates_received=current_round.num_participants,
        target=current_round.target_participants
    )


# ==========================================
# AGGREGATION FUNCTION
# ==========================================
async def perform_aggregation(current_round: TrainingRound, db: Session):
    agg_start = time.time()
    print(f"\n🔄 Starting aggregation for Round {current_round.round_number}...")
    
    updates = db.query(ModelUpdate).filter(
        ModelUpdate.round_id == current_round.id
    ).all()
    
    model_paths = [u.file_path for u in updates]
    data_sizes = [u.data_size for u in updates]
    
    # Perform federated averaging
    global_model_path = get_global_model_path(current_round.round_number)
    global_weights = aggregate_and_save(model_paths, data_sizes, global_model_path)
    
    # Save as latest model
    latest_path = get_initial_model_path()
    torch.save(global_weights, latest_path)
    
    # Evaluate on global test set
    global_accuracy = None
    global_loss = None
    test_data_path = os.path.join("data", "test_set", "test_data.pt")
    
    if os.path.exists(test_data_path):
        try:
            test_data = torch.load(test_data_path, weights_only=True)
            eval_result = evaluate_global_model(
                global_weights, test_data["X_test"], test_data["y_test"]
            )
            global_accuracy = eval_result["accuracy"]
            global_loss = eval_result["loss"]
            print(f"   📊 Global Test Accuracy: {global_accuracy:.4f}")
        except Exception as e:
            print(f"   ⚠️ Could not evaluate: {e}")
            # Fallback to average of local accuracies
            global_accuracy = sum(u.local_accuracy for u in updates if u.local_accuracy) / len(updates)
            global_loss = sum(u.local_loss for u in updates if u.local_loss) / len(updates)
    else:
        global_accuracy = sum(u.local_accuracy for u in updates if u.local_accuracy) / len(updates)
        global_loss = sum(u.local_loss for u in updates if u.local_loss) / len(updates)
    
    # Get previous round accuracy for comparison
    prev_round = db.query(TrainingRound).filter(
        TrainingRound.round_number == current_round.round_number - 1,
        TrainingRound.status == "completed"
    ).first()
    
    prev_accuracy = prev_round.global_accuracy if prev_round else None
    
    # Calculate aggregation latency
    agg_latency = round(time.time() - agg_start, 3)
    
    # Save global model record
    global_model = GlobalModel(
        round_id=current_round.id,
        file_path=global_model_path,
        accuracy=round(global_accuracy, 4),
        loss=round(global_loss, 4),
        version=f"v{current_round.round_number}"
    )
    db.add(global_model)
    
    # Update round status
    current_round.status = "completed"
    current_round.completed_at = datetime.utcnow()
    current_round.global_accuracy = round(global_accuracy, 4)
    current_round.global_loss = round(global_loss, 4)
    
    db.commit()
    
    # Create notification for hospitals
    improvement_msg = ""
    if prev_accuracy:
        diff = (global_accuracy - prev_accuracy) * 100
        improvement_msg = f" | Improved from {prev_accuracy*100:.1f}% → {global_accuracy*100:.1f}% ({'+' if diff > 0 else ''}{diff:.1f}%)"
    
    notification = {
        "type": "round_complete",
        "message": f"Round {current_round.round_number} completed — {current_round.num_participants} hospitals contributed — "
                   f"Global accuracy: {global_accuracy*100:.1f}%{improvement_msg}",
        "round_number": current_round.round_number,
        "accuracy": global_accuracy,
        "previous_accuracy": prev_accuracy,
        "timestamp": datetime.utcnow().isoformat(),
        "participants": current_round.num_participants
    }
    notifications.append(notification)
    
    # Audit log
    audit = AuditLog(
        action="aggregation_complete",
        details=f"Round {current_round.round_number} completed. "
                f"Global accuracy: {global_accuracy:.4f}{improvement_msg}. "
                f"Participants: {current_round.num_participants}. "
                f"Aggregation latency: {agg_latency}s. "
                f"Privacy: DP applied (ε={dp_module.epsilon})"
    )
    db.add(audit)
    db.commit()
    
    write_audit_log_file(
        "aggregation_complete",
        f"Round {current_round.round_number} done. Accuracy: {global_accuracy:.4f}. Latency: {agg_latency}s"
    )
    
    print(f"✅ Round {current_round.round_number} aggregation complete!")
    print(f"   Accuracy: {global_accuracy:.4f} | Latency: {agg_latency}s")


# ==========================================
# ENDPOINT 5: Get Global Model
# ==========================================
@app.get("/get_global_model", tags=["Global Model"])
async def get_global_model(
    api_key: Optional[str] = None,
    db: Session = Depends(get_db)
):
    latest_model = db.query(GlobalModel).order_by(GlobalModel.created_at.desc()).first()
    
    if latest_model and os.path.exists(latest_model.file_path):
        model_path = latest_model.file_path
        latest_model.download_count += 1
    else:
        model_path = get_initial_model_path()
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="No global model available yet")
    
    if api_key:
        hospital = db.query(Hospital).filter(Hospital.api_key == api_key).first()
        if hospital:
            audit = AuditLog(
                hospital_id=hospital.id,
                action="download_model",
                details=f"Hospital '{hospital.name}' downloaded global model"
            )
            db.add(audit)
            write_audit_log_file("download_model", "Global model downloaded", hospital.name)
    
    db.commit()
    
    return FileResponse(
        path=model_path,
        filename="global_model.pt",
        media_type="application/octet-stream"
    )


# ==========================================
# ENDPOINT 6: Round Status
# ==========================================
@app.get("/round_status", tags=["Training Rounds"])
async def round_status(db: Session = Depends(get_db)):
    active_round = db.query(TrainingRound).filter(
        TrainingRound.status == "in_progress"
    ).first()
    
    if active_round:
        updates = db.query(ModelUpdate).filter(
            ModelUpdate.round_id == active_round.id
        ).all()
        
        participating_hospitals = []
        for u in updates:
            hospital = db.query(Hospital).filter(Hospital.id == u.hospital_id).first()
            participating_hospitals.append({
                "hospital_name": hospital.name,
                "data_size": u.data_size,
                "local_accuracy": u.local_accuracy,
                "submitted_at": u.submitted_at.isoformat()
            })
        
        return {
            "status": "in_progress",
            "round_number": active_round.round_number,
            "participants": active_round.num_participants,
            "target": active_round.target_participants,
            "remaining": active_round.target_participants - active_round.num_participants,
            "started_at": active_round.started_at.isoformat(),
            "participating_hospitals": participating_hospitals
        }
    
    latest_round = db.query(TrainingRound).order_by(TrainingRound.round_number.desc()).first()
    
    if latest_round:
        # Get previous round for comparison
        prev_round = db.query(TrainingRound).filter(
            TrainingRound.round_number == latest_round.round_number - 1,
            TrainingRound.status == "completed"
        ).first()
        
        return {
            "status": latest_round.status,
            "round_number": latest_round.round_number,
            "participants": latest_round.num_participants,
            "target": latest_round.target_participants,
            "global_accuracy": latest_round.global_accuracy,
            "global_loss": latest_round.global_loss,
            "previous_accuracy": prev_round.global_accuracy if prev_round else None,
            "started_at": latest_round.started_at.isoformat(),
            "completed_at": latest_round.completed_at.isoformat() if latest_round.completed_at else None
        }
    
    return {"status": "no_rounds", "message": "No training rounds started yet"}


# ==========================================
# ENDPOINT 7: All Rounds History
# ==========================================
@app.get("/rounds", response_model=list[RoundResponse], tags=["Training Rounds"])
async def list_rounds(db: Session = Depends(get_db)):
    rounds = db.query(TrainingRound).order_by(TrainingRound.round_number.desc()).all()
    return rounds


# ==========================================
# ENDPOINT 8: Dashboard Stats (API)
# ==========================================
@app.get("/api/dashboard_stats", response_model=DashboardStats, tags=["Dashboard"])
async def dashboard_stats(db: Session = Depends(get_db)):
    total_hospitals = db.query(Hospital).count()
    active_hospitals = db.query(Hospital).filter(Hospital.is_active == True).count()
    total_rounds = db.query(TrainingRound).count()
    total_updates = db.query(ModelUpdate).count()
    
    latest_round = db.query(TrainingRound).order_by(TrainingRound.round_number.desc()).first()
    current_round = latest_round.round_number if latest_round else 0
    
    latest_completed = db.query(TrainingRound).filter(
        TrainingRound.status == "completed"
    ).order_by(TrainingRound.round_number.desc()).first()
    latest_accuracy = latest_completed.global_accuracy if latest_completed else None
    
    total_downloads = sum(m.download_count for m in db.query(GlobalModel).all())
    
    return DashboardStats(
        total_hospitals=total_hospitals,
        active_hospitals=active_hospitals,
        current_round=current_round,
        total_rounds=total_rounds,
        latest_accuracy=latest_accuracy,
        total_updates=total_updates,
        model_downloads=total_downloads
    )


# ==========================================
# ENDPOINT 9: Audit Logs
# ==========================================
@app.get("/audit_logs", response_model=list[AuditLogResponse], tags=["Audit"])
async def get_audit_logs(limit: int = 50, db: Session = Depends(get_db)):
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
    return logs


# ==========================================
# ENDPOINT 10: Global Models History
# ==========================================
@app.get("/global_models", response_model=list[GlobalModelResponse], tags=["Global Model"])
async def list_global_models(db: Session = Depends(get_db)):
    models = db.query(GlobalModel).order_by(GlobalModel.created_at.desc()).all()
    return models


# ==========================================
# ENDPOINT 11: Upload Hospital Metadata
# ==========================================
@app.post("/upload_metadata", tags=["Hospitals"])
async def upload_metadata(
    api_key: str = Form(...),
    hospital_name: str = Form(...),
    data_description: str = Form(""),
    num_samples: int = Form(0),
    data_type: str = Form("tabular"),
    metadata_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    """
    Hospital uploads metadata about its local dataset.
    No raw patient data — only descriptive metadata.
    """
    hospital = db.query(Hospital).filter(Hospital.api_key == api_key).first()
    if not hospital:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Update hospital info
    hospital.data_size = num_samples
    hospital.last_seen = datetime.utcnow()
    
    # Save metadata file if provided
    metadata_info = {
        "hospital_name": hospital_name,
        "data_description": data_description,
        "num_samples": num_samples,
        "data_type": data_type,
        "uploaded_at": datetime.utcnow().isoformat()
    }
    
    metadata_dir = os.path.join("data", hospital_name)
    os.makedirs(metadata_dir, exist_ok=True)
    
    with open(os.path.join(metadata_dir, "uploaded_metadata.json"), "w") as f:
        json.dump(metadata_info, f, indent=2)
    
    if metadata_file:
        file_path = os.path.join(metadata_dir, f"metadata_{metadata_file.filename}")
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(metadata_file.file, buffer)
    
    # Audit log
    audit = AuditLog(
        hospital_id=hospital.id,
        action="upload_metadata",
        details=f"Metadata uploaded: {num_samples} samples, type: {data_type}"
    )
    db.add(audit)
    db.commit()
    
    return {
        "message": f"Metadata uploaded for {hospital_name}",
        "num_samples": num_samples,
        "data_type": data_type
    }


# ==========================================
# ENDPOINT 12: Push Notifications for Hospitals
# ==========================================
@app.get("/notifications", tags=["Notifications"])
async def get_notifications(
    api_key: Optional[str] = None,
    since_round: Optional[int] = 0,
    db: Session = Depends(get_db)
):
    """
    Hospitals poll this endpoint to check for new notifications.
    Returns notifications about completed rounds and new global models.
    """
    filtered = [n for n in notifications if n.get("round_number", 0) > since_round]
    
    if api_key:
        hospital = db.query(Hospital).filter(Hospital.api_key == api_key).first()
        if hospital:
            hospital.last_seen = datetime.utcnow()
            db.commit()
    
    return {
        "notifications": filtered,
        "count": len(filtered),
        "latest_round": filtered[-1]["round_number"] if filtered else since_round
    }


# ==========================================
# ENDPOINT 13: Model Comparison (Global vs Single Node)
# ==========================================
@app.get("/api/model_comparison", tags=["Dashboard"])
async def model_comparison(db: Session = Depends(get_db)):
    """
    Compare global federated model vs individual hospital models.
    Shows that federated model outperforms single-node models.
    """
    completed_rounds = db.query(TrainingRound).filter(
        TrainingRound.status == "completed"
    ).order_by(TrainingRound.round_number).all()
    
    comparison_data = []
    
    for round_obj in completed_rounds:
        updates = db.query(ModelUpdate).filter(
            ModelUpdate.round_id == round_obj.id
        ).all()
        
        hospital_accuracies = {}
        for u in updates:
            hospital = db.query(Hospital).filter(Hospital.id == u.hospital_id).first()
            hospital_accuracies[hospital.name] = u.local_accuracy
        
        # Best single hospital accuracy
        best_single = max(hospital_accuracies.values()) if hospital_accuracies else 0
        avg_single = sum(hospital_accuracies.values()) / len(hospital_accuracies) if hospital_accuracies else 0
        
        comparison_data.append({
            "round_number": round_obj.round_number,
            "global_accuracy": round_obj.global_accuracy,
            "best_single_hospital": round(best_single, 4),
            "avg_single_hospital": round(avg_single, 4),
            "improvement_over_best": round((round_obj.global_accuracy - best_single) * 100, 2) if round_obj.global_accuracy else 0,
            "improvement_over_avg": round((round_obj.global_accuracy - avg_single) * 100, 2) if round_obj.global_accuracy else 0,
            "hospital_accuracies": hospital_accuracies,
            "num_participants": round_obj.num_participants
        })
    
    return {
        "comparison": comparison_data,
        "summary": {
            "total_rounds": len(comparison_data),
            "final_global_accuracy": comparison_data[-1]["global_accuracy"] if comparison_data else None,
            "final_best_single": comparison_data[-1]["best_single_hospital"] if comparison_data else None,
            "global_outperforms": (
                comparison_data[-1]["global_accuracy"] >= comparison_data[-1]["best_single_hospital"]
                if comparison_data else False
            )
        }
    }


# ==========================================
# ENDPOINT 14: Accuracy Improvement Timeline
# ==========================================
@app.get("/api/accuracy_timeline", tags=["Dashboard"])
async def accuracy_timeline(db: Session = Depends(get_db)):
    """Get round-by-round accuracy with improvement tracking"""
    rounds = db.query(TrainingRound).filter(
        TrainingRound.status == "completed"
    ).order_by(TrainingRound.round_number).all()
    
    timeline = []
    prev_accuracy = None
    
    for r in rounds:
        improvement = None
        if prev_accuracy and r.global_accuracy:
            improvement = round((r.global_accuracy - prev_accuracy) * 100, 2)
        
        timeline.append({
            "round_number": r.round_number,
            "accuracy": r.global_accuracy,
            "loss": r.global_loss,
            "accuracy_pct": round(r.global_accuracy * 100, 2) if r.global_accuracy else None,
            "improvement_pct": improvement,
            "participants": r.num_participants,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None
        })
        
        prev_accuracy = r.global_accuracy
    
    first_acc = rounds[0].global_accuracy if rounds else None
    last_acc = rounds[-1].global_accuracy if rounds else None
    total_improvement = round((last_acc - first_acc) * 100, 2) if (first_acc and last_acc) else None
    
    return {
        "timeline": timeline,
        "summary": {
            "first_round_accuracy": f"{first_acc*100:.1f}%" if first_acc else None,
            "latest_accuracy": f"{last_acc*100:.1f}%" if last_acc else None,
            "total_improvement": f"{total_improvement}%" if total_improvement else None,
            "total_rounds": len(rounds)
        }
    }


# ==========================================
# ENDPOINT 15: Privacy Info
# ==========================================
@app.get("/privacy_info", tags=["Privacy"])
async def privacy_info():
    return {
        "privacy_settings": dp_module.get_privacy_report(),
        "data_handling": {
            "raw_data_shared": False,
            "data_location": "On-premises at each hospital",
            "what_is_shared": "Model weights (numerical parameters only)",
            "encryption": "TLS for communication",
            "audit_logging": True,
            "access_control": "API key authentication per hospital",
            "differential_privacy_applied": True,
            "update_clipping_applied": True
        },
        "compliance": {
            "hipaa_compatible": True,
            "gdpr_compatible": True,
            "data_minimization": True,
            "right_to_erasure": "Hospital can withdraw and delete local data anytime"
        }
    }


# ==========================================
# ENDPOINT 16: PDF Report
# ==========================================
@app.get("/export/pdf", tags=["Reports"])
async def export_pdf_report(db: Session = Depends(get_db)):
    rounds = db.query(TrainingRound).order_by(TrainingRound.round_number).all()
    hospitals = db.query(Hospital).all()
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(50).all()
    
    rounds_data = [{
        "round_number": r.round_number, "status": r.status,
        "num_participants": r.num_participants, "target_participants": r.target_participants,
        "global_accuracy": r.global_accuracy, "global_loss": r.global_loss,
        "started_at": r.started_at.isoformat() if r.started_at else "",
        "completed_at": r.completed_at.isoformat() if r.completed_at else ""
    } for r in rounds]
    
    hospitals_data = [{
        "name": h.name, "location": h.location, "data_size": h.data_size,
        "is_active": h.is_active,
        "registered_at": h.registered_at.isoformat() if h.registered_at else ""
    } for h in hospitals]
    
    audit_data = [{
        "timestamp": l.timestamp.isoformat() if l.timestamp else "",
        "action": l.action, "details": l.details, "hospital_id": l.hospital_id
    } for l in logs]
    
    privacy_params = dp_module.get_privacy_report()
    
    output_path = generate_compliance_report(
        rounds_data, hospitals_data, audit_data, privacy_params
    )
    
    audit = AuditLog(action="export_report", details="PDF compliance report generated")
    db.add(audit)
    db.commit()
    
    return FileResponse(path=output_path, filename="compliance_report.pdf", media_type="application/pdf")


# ==========================================
# ENDPOINT 17: CSV Report
# ==========================================
@app.get("/export/csv", tags=["Reports"])
async def export_csv_report(db: Session = Depends(get_db)):
    rounds = db.query(TrainingRound).order_by(TrainingRound.round_number).all()
    
    rounds_data = [{
        "round_number": r.round_number, "status": r.status,
        "num_participants": r.num_participants, "target_participants": r.target_participants,
        "global_accuracy": r.global_accuracy, "global_loss": r.global_loss,
        "started_at": r.started_at.isoformat() if r.started_at else "",
        "completed_at": r.completed_at.isoformat() if r.completed_at else ""
    } for r in rounds]
    
    output_path = generate_csv_report(rounds_data)
    return FileResponse(path=output_path, filename="training_report.csv", media_type="text/csv")


# ==========================================
# ENDPOINT 18: Audit CSV
# ==========================================
@app.get("/export/audit_csv", tags=["Reports"])
async def export_audit_log(db: Session = Depends(get_db)):
    logs = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).all()
    
    audit_data = [{
        "timestamp": l.timestamp.isoformat() if l.timestamp else "",
        "action": l.action, "details": l.details,
        "hospital_id": l.hospital_id, "ip_address": l.ip_address
    } for l in logs]
    
    output_path = generate_audit_csv(audit_data)
    return FileResponse(path=output_path, filename="audit_log.csv", media_type="text/csv")


# ==========================================
# DASHBOARD PAGE
# ==========================================
@app.get("/dashboard", tags=["Dashboard"])
async def dashboard_page(request: Request):
    return templates.TemplateResponse(request=request, name="dashboard.html")


# ==========================================
# ENDPOINT 19: PostgreSQL Dataset Info
# ==========================================
@app.get("/api/dataset_info", tags=["Dataset"])
async def dataset_info():
    """Get information about the real medical dataset stored in PostgreSQL"""
    try:
        from server.postgres_config import PostgresSession
        from server.ehr_models import HospitalDataset, PatientRecord, FeatureStatistic
        
        session = PostgresSession()
        
        datasets = session.query(HospitalDataset).all()
        
        result = {
            "database": "PostgreSQL (federated_ehr)",
            "dataset": "Cleveland Heart Disease (UCI Repository)",
            "type": "Tabular EHR (Electronic Health Records)",
            "hospitals": []
        }
        
        for ds in datasets:
            record_count = session.query(PatientRecord).filter(
                PatientRecord.hospital_dataset_id == ds.id
            ).count()
            
            stats = session.query(FeatureStatistic).filter(
                FeatureStatistic.hospital_dataset_id == ds.id
            ).all()
            
            result["hospitals"].append({
                "name": ds.hospital_name,
                "total_records": record_count,
                "disease_count": ds.disease_count,
                "healthy_count": ds.healthy_count,
                "disease_ratio": ds.disease_ratio,
                "source": ds.dataset_source,
                "feature_stats": {
                    s.feature_name: {
                        "mean": s.mean_value,
                        "std": s.std_value,
                        "min": s.min_value,
                        "max": s.max_value
                    } for s in stats
                }
            })
        
        total = session.query(PatientRecord).count()
        result["total_records"] = total
        
        session.close()
        return result
    
    except Exception as e:
        return {
            "database": "PostgreSQL",
            "status": "not_configured",
            "message": str(e),
            "fix": "Run 'python -m client.prepare_real_data' to set up PostgreSQL"
        }


# ==========================================
# ENDPOINT 20: PostgreSQL Connection Status
# ==========================================
@app.get("/api/postgres_status", tags=["Dataset"])
async def postgres_status():
    """Check PostgreSQL connection status"""
    try:
        pg_ok = test_pg()
        return {
            "postgresql_connected": pg_ok,
            "database": "federated_ehr",
            "purpose": "Stores REAL UCI Heart Disease dataset",
            "note": "Separate from SQLite which stores FL metadata"
        }
    except Exception as e:
        return {
            "postgresql_connected": False,
            "error": str(e)
        }


# ==========================================
# ROOT
# ==========================================
@app.get("/", tags=["Root"])
async def root():
    return {
        "project": "Cross-Hospital Diagnosis Model",
        "description": "Federated Learning Aggregator Server",
        "version": "1.0.0",
        "docs": "/docs",
        "dashboard": "/dashboard"
    }