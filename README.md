# 🏥 Cross-Hospital Diagnosis Model

## Federated Learning System for Collaborative Medical Diagnosis

A privacy-preserving federated learning system that enables multiple hospitals to collaboratively train diagnostic models **without sharing raw patient data**. Each hospital trains locally and sends only **model weights** to a central **FastAPI aggregator** that performs **Federated Averaging (FedAvg)**, tracks rounds, provides dashboards, audit logs, and compliance exports.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Endpoints](#api-endpoints)
- [Dashboard](#dashboard)
- [Privacy & Security](#privacy--security)
- [How Federated Learning Works](#how-federated-learning-works)
- [Limitations](#limitations)
- [References](#references)

---

## 🎯 Overview

In centralized ML, hospitals would need to share patient data for training, which is typically blocked by privacy regulations (HIPAA/GDPR) and security concerns. This project demonstrates a **federated learning** workflow where:

- Hospitals keep patient data on-premises
- Hospitals train a local copy of the model
- Only **weights** (numerical parameters) are uploaded
- The server aggregates weights into an improved global model
- A dashboard shows round progress, metrics, and audit history

Supported diagnosis modalities (separate demos):
- **Type 1 (Tabular EHR):** Heart disease risk prediction (MLP)
- **Type 2 (Medical Imaging):** Chest X-ray classification (CNN) *(simulated images)*
- **Type 3 (Clinical Text):** Disease category from clinical notes (BoW + MLP) *(simulated notes)*

### Key Benefits:
- ✅ **Privacy-Preserving**: No raw patient data leaves the hospital
- ✅ **Collaborative**: Multiple hospitals improve the model together
- ✅ **Regulatory Compliant**: HIPAA/GDPR compatible by design
- ✅ **Scalable**: Add new hospitals without data migration

---

## 🏗️ Architecture


┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│ Hospital A │ │ Hospital B │ │ Hospital C │
│ (New York) │ │ (London) │ │ (Tokyo) │
│ │ │ │ │ │
│ Local Data │ │ Local Data │ │ Local Data │
│ 300 records │ │ 300 records │ │ 300 records │
│ │ │ │ │ │
│ Train Model │ │ Train Model │ │ Train Model │
│ Locally │ │ Locally │ │ Locally │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
│ │ │
│ Weights Only │ Weights Only │ Weights Only
│ (No Patient │ (No Patient │ (No Patient
│ Data!) │ Data!) │ Data!)
▼ ▼ ▼
┌─────────────────────────────────────────────────────────┐
│ AGGREGATOR SERVER (FastAPI) │
│ │
│ • Collects model weight updates from hospitals │
│ • Performs Federated Averaging (FedAvg) │
│ • Distributes improved global model │
│ • Dashboard for monitoring & compliance │
│ • Generates audit logs & reports │
└─────────────────────────────────────────────────────────┘

Key components:
- **Hospital Nodes (Clients)**: local training + weight submission
- **Aggregator Server (FastAPI)**: orchestrates rounds and aggregation
- **SQLite**: federated metadata (hospitals, rounds, updates, audit logs)
- **PostgreSQL**: real dataset storage for tabular EHR simulation (UCI Heart Disease)
- **Dashboard UI**: monitoring + exports (PDF/CSV)

---

## ✨ Features

### Core Features
- **Federated Averaging (FedAvg)**: Weighted aggregation of model updates
- **Multi-Hospital Support**: Register and manage multiple hospital nodes
- **Round Management**: Start, monitor, and complete training rounds
- **Model Versioning**: Track global model versions across rounds

### Dashboard & Monitoring
- **Real-time Dashboard**: Web UI showing round progress, accuracy, participation
- **Accuracy Charts**: Visual tracking of model improvement over rounds
- **Hospital Management**: View registered hospitals and their status
- **Activity Feed**: Real-time audit log of all system actions
- **Notifications**: Clients can poll `/notifications` when a new global model is ready
- **Metadata**: Metadata upload UI for hospitals (safe “data about data”)

### Privacy & Security
- **Differential Privacy**: Calibrated noise added to model updates
- **Update Clipping**: Gradient clipping to limit sensitivity
- **API Key Authentication**: Each hospital has a unique API key
- **Audit Logging**: Complete trail of all actions for compliance

### Reporting
- **PDF Compliance Report**: Professional report for regulators
- **CSV Export**: Training data and audit logs exportable as CSV
- **Privacy Information**: Detailed privacy settings and guarantees

### Dataset Support
- **Real public dataset**: UCI Heart Disease (tabular EHR-style), stored in PostgreSQL
- Dataset partitioned into simulated hospitals
- Exported into tensor `.pt` files for training compatibility

### Two databases
- SQLite: federated system metadata
- PostgreSQL: real dataset storage (UCI Heart Disease)

### Supports multiple diagnosis modalities:
- **Tabular EHR** (MLP)
- **Medical images** (CNN)
- **Clinical text** (Bag-of-Words + MLP)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backend/API** | Python + FastAPI |
| **ML Framework** | PyTorch |
| **Database** | SQLite + SQLAlchemy + PostgreSQL |
| **Frontend** | HTML5, CSS3, JavaScript |
| **Charts** | Chart.js |
| **PDF Reports** | ReportLab |
| **Data Processing** | Pandas, NumPy, Scikit-learn |

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd cross_hospital_diagnosis

2. **Create virtual environment**
python -m venv venv
source venv/Scripts/activate    # Windows (Git Bash)
# source venv/bin/activate      # Mac/Linux

3. **Install dependencies**
pip install -r requirements.txt

If torch fails:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Start the Server
```bash
source venv/Scripts/activate
python -m uvicorn server.main:app --reload --port 8000

Open:

Dashboard: http://127.0.0.1:8000/dashboard
Swagger docs: http://127.0.0.1:8000/docs
```
## 🐘 PostgreSQL Setup (Real UCI Dataset)

**Create database:**
```SQL
CREATE DATABASE federated_ehr;
```

**Load real dataset into PostgreSQL:**
```bash
python -m client.prepare_real_data


**Export PostgreSQL partitions into .pt files**
python -m client.load_from_postgres

**Run Tabular Federated Simulation (3 hospitals, 3 rounds)**
python -m client.run_simulation

**Run Image + Text Diagnosis Demos (Optional / Separate)**
Image (CNN):
python -m client.prepare_image_data
python -m client.image_client

Text (BoW + MLP):
python -m client.prepare_text_data
python -m client.text_client

Run all three demos:
python -m client.run_all_demos

```

---

## 📁 Project Structure (Core)

### server/
  main.py                # FastAPI aggregator + dashboard routes
  database.py            # SQLite metadata DB config
  models.py              # SQLite tables: hospitals, rounds, updates, etc.
  postgres_config.py     # PostgreSQL config (real dataset)
  ehr_models.py          # PostgreSQL tables: patient_records, etc.
  model_definition.py    # Tabular MLP (PyTorch)
  image_model.py         # Image CNN (PyTorch)
  text_model.py          # Text model (PyTorch)
  aggregation.py         # FedAvg implementation
  privacy.py             # DP (clip+noise) + integrity validation
  reports.py             # PDF/CSV report generation
  templates/dashboard.html
  static/css/style.css
  static/js/dashboard.js

### client/
  run_simulation.py      # Runs federated rounds (tabular)
  hospital_client.py     # Hospital node: download/train/submit update
  prepare_real_data.py   # UCI Heart Disease → PostgreSQL
  load_from_postgres.py  # PostgreSQL → .pt tensors (compatible training)
  prepare_image_data.py  # Simulated X-ray dataset
  image_client.py        # Local CNN demo
  prepare_text_data.py   # Simulated clinical notes dataset
  text_client.py         # Local text model demo
  run_all_demos.py       # Runs tabular+image+text demos

---

## API Endpoints

**Core (Aggregator):**

- POST /register_hospital
- POST /start_round
- POST /submit_update
- GET /get_global_model
- GET /round_status

**Monitoring:**

- GET /dashboard
- GET /api/dashboard_stats
- GET /api/accuracy_timeline
- GET /api/model_comparison
- GET /audit_logs
- GET /notifications

**Reports:**

- GET /export/pdf
- GET /export/csv
- GET /export/audit_csv

**Dataset:**

- GET /api/postgres_status
- GET /api/dataset_info

---

## Dashboard
### Dashboard URL: http://127.0.0.1:8000/dashboard

**Shows:**

- Round status banner (“Round X completed — N hospitals — accuracy improved …”)
- Statistics cards
- Accuracy trend chart
- Global vs single hospital comparison chart
- Round history, hospital list, model versions
- Audit log feed
- Export buttons (PDF/CSV)
- Metadata upload form (safe metadata only)

---
## Privacy & Security
**This system ensures:**

- No raw patient data leaves hospital nodes
- Only weights are exchanged
- Differential Privacy is applied to updates (clip + noise)
- Audit logs support compliance review
- API key authentication restricts submissions to registered hospitals

---

## How Federated Learning Works
- Server initializes a global model.
- Hospitals download the model.
- Each hospital trains on its local dataset.
- Each hospital uploads only model weights to the server.
- Server aggregates updates using FedAvg:
- W_global = Σ (n_i / N) * W_i
- Server saves and publishes the new global model.
- Repeat for multiple rounds to improve accuracy.

---

## Limitations
- Non-IID data across hospitals can slow convergence or reduce accuracy
- Requires multiple rounds (often more than centralized training)
- Client dropouts can delay aggregation (future: async aggregation, timeouts)
- Secure aggregation / HE encryption is not fully implemented (can be extended)

---

## References
### Federated Learning Survey (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC7523633/
### Recent Advances Survey (arXiv): https://arxiv.org/abs/2301.01299
### UCI Heart Disease Dataset: https://archive.ics.uci.edu/ml/datasets/Heart+Disease