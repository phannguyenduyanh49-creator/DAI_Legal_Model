import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base


DATABASE_URL = os.getenv(
    "CATALOG_DATABASE_URL",
    "postgresql+psycopg2://legal_user:legal_password@localhost:5432/legal_ai_db",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()