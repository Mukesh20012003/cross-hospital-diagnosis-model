# server/utils.py

import secrets
import hashlib
import os
import json
from datetime import datetime


def generate_api_key():
    """Generate a unique API key for hospital authentication"""
    raw_key = secrets.token_hex(32)
    return f"hosp_{raw_key}"


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage (optional extra security)"""
    return hashlib.sha256(api_key.encode()).hexdigest()


def get_model_update_path(round_number: int, hospital_id: int) -> str:
    """Generate file path for saving hospital model updates"""
    directory = os.path.join("models", "updates", f"round_{round_number}")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"hospital_{hospital_id}_weights.pt")


def get_global_model_path(round_number: int) -> str:
    """Generate file path for saving global model"""
    directory = os.path.join("models", "global")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"global_model_round_{round_number}.pt")


def get_initial_model_path() -> str:
    """Path for the initial global model"""
    directory = os.path.join("models", "global")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "initial_model.pt")


def write_audit_log_file(action: str, details: str, hospital_name: str = None):
    """Write audit log to file for compliance"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"audit_{datetime.utcnow().strftime('%Y%m%d')}.log")
    
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "hospital": hospital_name,
        "details": details
    }
    
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")