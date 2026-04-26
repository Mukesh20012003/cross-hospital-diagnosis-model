# server/postgres_config.py

"""
PostgreSQL Configuration for Real Medical Dataset Storage
==========================================================
This database stores the REAL Heart Disease UCI dataset.
Separate from SQLite (which stores FL metadata like rounds, hospitals, etc.)

SQLite  → Federated Learning metadata (rounds, hospitals, audit logs)
PostgreSQL → Real medical/EHR dataset (patient records, clinical features)
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ============================================
# UPDATE THESE WITH YOUR POSTGRESQL CREDENTIALS
# ============================================
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "12345"   # ← CHANGE THIS
POSTGRES_HOST = "localhost"
POSTGRES_PORT = "5433"
POSTGRES_DB = "federated_ehr"

POSTGRES_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Create PostgreSQL engine
postgres_engine = create_engine(POSTGRES_URL, echo=False)

# Session factory
PostgresSession = sessionmaker(autocommit=False, autoflush=False, bind=postgres_engine)

# Base for PostgreSQL models
PostgresBase = declarative_base()


def get_postgres_db():
    """Get PostgreSQL database session"""
    db = PostgresSession()
    try:
        yield db
    finally:
        db.close()


def test_connection():
    """Test if PostgreSQL connection works"""
    try:
        conn = postgres_engine.connect()
        conn.close()
        print("✅ PostgreSQL connection successful!")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print(f"   Make sure PostgreSQL is running and credentials are correct.")
        print(f"   URL: {POSTGRES_URL}")
        return False