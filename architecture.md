# 🏗️ Architecture Diagram

## System Overview

┌─────────────────────────────────────────────────────────────────────────┐
│ CROSS-HOSPITAL DIAGNOSIS MODEL │
│ Federated Learning Architecture │
└─────────────────────────────────────────────────────────────────────────┘

┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐
│ HOSPITAL A │ │ HOSPITAL B │ │ HOSPITAL C │
│ (New York) │ │ (London) │ │ (Tokyo) │
│ │ │ │ │ │
│ ┌────────────────┐ │ │ ┌────────────────┐ │ │ ┌────────────────┐ │
│ │ PostgreSQL │ │ │ │ PostgreSQL │ │ │ │ PostgreSQL │ │
│ │ (Local EHR) │ │ │ │ (Local EHR) │ │ │ │ (Local EHR) │ │
│ │ 90 patients │ │ │ │ 90 patients │ │ │ │ 90 patients │ │
│ └───────┬────────┘ │ │ └───────┬────────┘ │ │ └───────┬────────┘ │
│ │ │ │ │ │ │ │ │
│ ▼ │ │ ▼ │ │ ▼ │
│ ┌────────────────┐ │ │ ┌────────────────┐ │ │ ┌────────────────┐ │
│ │ EHR Preprocess │ │ │ │ EHR Preprocess │ │ │ │ EHR Preprocess │ │
│ │ StandardScaler │ │ │ │ StandardScaler │ │ │ │ StandardScaler │ │
│ └───────┬────────┘ │ │ └───────┬────────┘ │ │ └───────┬────────┘ │
│ │ │ │ │ │ │ │ │
│ ▼ │ │ ▼ │ │ ▼ │
│ ┌────────────────┐ │ │ ┌────────────────┐ │ │ ┌────────────────┐ │
│ │ PyTorch │ │ │ │ PyTorch │ │ │ │ PyTorch │ │
│ │ Local Train │ │ │ │ Local Train │ │ │ │ Local Train │ │
│ │ (5 epochs) │ │ │ │ (5 epochs) │ │ │ │ (5 epochs) │ │
│ │ nn.Module │ │ │ │ nn.Module │ │ │ │ nn.Module │ │
│ │ Adam + CE │ │ │ │ Adam + CE │ │ │ │ Adam + CE │ │
│ └───────┬────────┘ │ │ └───────┬────────┘ │ │ └───────┬────────┘ │
│ │ │ │ │ │ │ │ │
│ WEIGHTS ONLY │ │ WEIGHTS ONLY │ │ WEIGHTS ONLY │
│ (No patient │ │ (No patient │ │ (No patient │
│ data sent!) │ │ data sent!) │ │ data sent!) │
│ │ │ │ │ │ │ │ │
└─────────┼──────────┘ └─────────┼──────────┘ └─────────┼──────────┘
│ │ │
│ POST /submit_update │ │
│ (API Key Auth) │ │
▼ ▼ ▼
┌─────────────────────────────────────────────────────────────────────┐
│ │
│ SECURE CHANNEL (TLS/HTTPS) │
│ │
└───────────────────────────┬─────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│ │
│ AGGREGATOR SERVER (FastAPI + Python) │
│ │
│ ┌──────────────────────────────────────────────────────────────┐ │
│ │ API ENDPOINTS │ │
│ │ POST /register_hospital POST /submit_update │ │
│ │ POST /start_round GET /get_global_model │ │
│ │ GET /round_status GET /hospitals │ │
│ │ GET /rounds GET /audit_logs │ │
│ │ GET /notifications POST /upload_metadata │ │
│ │ GET /export/pdf GET /export/csv │ │
│ │ GET /api/model_comparison GET /api/accuracy_timeline │ │
│ │ GET /api/dataset_info GET /privacy_info │ │
│ └──────────────────────────────────────────────────────────────┘ │
│ │ │
│ ┌───────────────┼───────────────┐ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ │
│ │ FedAvg │ │ Differential │ │ Update │ │
│ │ Aggregation │ │ Privacy │ │ Validation │ │
│ │ (Weighted │ │ (ε=1.0) │ │ (NaN/Inf check) │ │
│ │ Average) │ │ Clip + Noise │ │ │ │
│ └──────┬───────┘ └──────────────┘ └──────────────────┘ │
│ │ │
│ ▼ │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ DATABASES │ │
│ │ │ │
│ │ SQLite (federated_learning.db) │ │
│ │ ├── hospitals, training_rounds │ │
│ │ ├── model_updates, global_models │ │
│ │ └── audit_logs │ │
│ │ │ │
│ │ PostgreSQL (federated_ehr) │ │
│ │ ├── hospital_datasets │ │
│ │ ├── patient_records (Real UCI Heart Disease) │ │
│ │ └── feature_statistics │ │
│ └──────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ MODEL STORAGE (File System) │ │
│ │ ├── models/global/initial_model.pt │ │
│ │ ├── models/global/global_model_round_X.pt │ │
│ │ └── models/updates/round_X/hospital_Y_weights.pt │ │
│ └──────────────────────────────────────────────────────────┘ │
│ │
└─────────────────────────────────┬──────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────────┐
│ │
│ WEB DASHBOARD (Browser) │
│ │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │
│ │ Stats Cards│ │ Accuracy │ │ Global vs │ │ Audit │ │
│ │ Hospitals │ │ Chart │ │ Single │ │ Logs │ │
│ │ Rounds │ │ (Chart.js) │ │ Comparison │ │ │ │
│ │ Accuracy │ │ │ │ Bar Chart │ │ │ │
│ └────────────┘ └────────────┘ └────────────┘ └────────────┘ │
│ │
│ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ │
│ │ Round │ │ Hospital │ │ Model │ │ Metadata │ │
│ │ History │ │ Table │ │ Versions │ │ Upload │ │
│ │ Table │ │ │ │ Table │ │ Form │ │
│ └────────────┘ └────────────┘ └────────────┘ └────────────┘ │
│ │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ EXPORT: PDF Compliance Report | CSV Training | Audit CSV│ │
│ └──────────────────────────────────────────────────────────┘ │
│ │
└─────────────────────────────────────────────────────────────────────┘



## Data Flow

ROUND N FLOW:
═════════════

Server: POST /start_round
└── Creates round N, status = "in_progress"

Each Hospital:
├── GET /get_global_model (download latest weights)
├── Load LOCAL data from PostgreSQL / .pt file
├── PyTorch training: model.train() for 5 epochs
│ ├── Forward: outputs = model(patient_features)
│ ├── Loss: CrossEntropyLoss(outputs, diagnosis)
│ ├── Backward: loss.backward()
│ └── Update: optimizer.step()
├── Evaluate on validation set
└── POST /submit_update (sends weights + accuracy)
├── Server validates update integrity
├── Differential Privacy: clip + add noise
└── Save weights to models/updates/

Server (when all hospitals submitted):
├── FedAvg: Global_W = Σ (n_i/N) × W_i
├── Evaluate on global test set
├── Save global model: models/global/
├── Update round status = "completed"
├── Send notification to hospitals
└── Log to audit trail

Dashboard updates automatically (10s refresh)


## Privacy Architecture

┌─────────────────────────────────────────────────┐
│ PRIVACY-PRESERVING PIPELINE │
├─────────────────────────────────────────────────┤
│ │
│ Hospital trains locally │
│ │ │
│ ▼ │
│ Raw weights: [0.523, -0.891, 0.234, ...] │
│ │ │
│ ▼ │
│ Step 1: CLIP (max_grad_norm = 1.0) │
│ Clipped: [0.480, -0.817, 0.215, ...] │
│ │ │
│ ▼ │
│ Step 2: ADD NOISE (Gaussian, ε=1.0) │
│ Noisy: [0.502, -0.843, 0.198, ...] │
│ │ │
│ ▼ │
│ Step 3: VALIDATE (check NaN, Inf, magnitude) │
│ │ │
│ ▼ │
│ Sent to server (ONLY these numbers) │
│ │
│ ❌ Patient names → NOT sent │
│ ❌ Patient records → NOT sent │
│ ❌ Raw features → NOT sent │
│ ✅ Model weights → Clipped + Noised + Sent │
│ │
└─────────────────────────────────────────────────┘

