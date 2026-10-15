"""Подключение к SQLite и отдельная сессия для каждого запроса."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{Path(__file__).resolve().parents[2] / 'agro_advisor.db'}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Выдаём сессию запроса и закрываем её даже при ошибке."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
