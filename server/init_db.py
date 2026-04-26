# server/init_db.py

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.database import engine, Base
from server.models import Hospital, TrainingRound, ModelUpdate, GlobalModel, AuditLog


def initialize_database():
    """Create all database tables"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✅ All tables created successfully!")
    print("\nTables created:")
    print("  - hospitals")
    print("  - training_rounds")
    print("  - model_updates")
    print("  - global_models")
    print("  - audit_logs")


if __name__ == "__main__":
    initialize_database()